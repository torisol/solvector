# solvector/evaluate.py
import argparse, json, math, random, re
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"\w+|\S")
def tokenize(t): return [x.lower() for x in TOKEN_RE.findall(t or "")]

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_user_text(obj):
    if "user_text" in obj and isinstance(obj["user_text"], str): return obj["user_text"]
    u = obj.get("user") or {}
    if isinstance(u, dict): return u.get("text") or u.get("content") or ""
    if isinstance(obj.get("user"), str): return obj["user"]
    return ""

def make_head(in_dim, y_dim, dropout=0.0, bottleneck=0):
    if bottleneck and bottleneck > 0:
        return nn.Sequential(
            nn.Linear(in_dim, bottleneck), nn.ReLU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(bottleneck, y_dim),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(in_dim, y_dim),
        )

class GRURegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, y_dim, pad_idx, dropout=0.0, bottleneck=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = make_head(hidden_dim, y_dim, dropout=dropout, bottleneck=bottleneck)
        self.pad_idx = pad_idx
    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout); self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        if self.batch_first: x = x + self.pe[:, :x.size(1), :]
        else:                 x = x + self.pe[:, :x.size(0), :].transpose(0,1)
        return self.dropout(x)

class AttnPool(nn.Module):
    def __init__(self, dim): super().__init__(); self.q = nn.Parameter(torch.randn(dim))
    def forward(self, h, mask):
        q = self.q.view(1,1,-1)
        scores = (h * q).sum(dim=-1).masked_fill(mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (w * h).sum(dim=1)

class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, y_dim, pad_idx,
                 nheads=4, nlayers=2, ffn_dim=512, pe_dropout=0.1,
                 head_dropout=0.0, pool="mean", bottleneck=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pe  = PositionalEncoding(embed_dim, dropout=pe_dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads,
                                           dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = make_head(embed_dim, y_dim, dropout=head_dropout, bottleneck=bottleneck)
        self.pad_idx = pad_idx
        self.pool = pool
        if pool == "attn": self.attnpool = AttnPool(embed_dim)
    def forward(self, x, lengths):
        mask = (x == self.pad_idx)
        h = self.emb(x); h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        if self.pool == "cls":
            z = h[:,0,:]
        elif self.pool == "mean":
            valid = (x != self.pad_idx)
            if valid.size(1) > 0: valid[:,0] = False
            denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1).float()
            z = (h.masked_fill((~valid).unsqueeze(-1), 0.0).sum(dim=1)) / denom
        else:
            z = self.attnpool(h, mask)
        return self.head(z)

def encode(text, stoi, max_tokens, unk_idx=1, empty_idx=None, cls_idx=None, use_cls=True):
    toks = tokenize(text)
    if not toks and empty_idx is not None: toks = ["<empty>"]
    ids = [stoi.get(t, unk_idx) for t in toks[:max_tokens]]
    if use_cls and (cls_idx is not None): ids = [cls_idx] + ids
    if not ids: ids = [unk_idx]
    X = torch.tensor(ids, dtype=torch.long)
    return X, X.numel()

def cosine(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="all", choices=["all","val","train"])
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-norm", action="store_true")
    ap.add_argument("--dump-bottomk", type=int, default=0)
    ap.add_argument("--dump-topk", type=int, default=0)
    ap.add_argument("--dump-path", type=str, default="out/global.jsonl")
    ap.add_argument("--bucket-dump-bottomk", type=int, default=0)
    ap.add_argument("--bucket-dump-topk", type=int, default=0)
    ap.add_argument("--bucket-dump-path", type=str, default="out/by_bucket.jsonl")
    ap.add_argument("--eps", type=float, default=1e-8)
    args = ap.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(Path(args.ckpt), map_location=DEVICE)
    cfg  = blob["config"]
    stoi = blob["stoi"]; itos = blob["itos"]

    enc     = cfg.get("ENCODER","gru")
    pool    = cfg.get("POOL","mean")
    ydim    = cfg["Y_DIM"]
    pad_idx = cfg["PAD_IDX"]
    max_tokens = cfg["MAX_TOKENS"]
    bottleneck = cfg.get("BOTTLENECK", 0)

    print(f"[ckpt] ENCODER={enc}  POOL={pool if enc!='gru' else '-'}  EMBED={cfg.get('EMBED_DIM','-')}  "
          f"NLAYERS={cfg.get('NLAYERS','-')}  NHEADS={cfg.get('NHEADS','-')}  BOTTLENECK={bottleneck}")

    if enc == "gru":
        model = GRURegressor(len(itos), cfg["EMBED_DIM"], cfg.get("HIDDEN_DIM",256), ydim, pad_idx,
                             dropout=cfg.get("DROPOUT",0.0), bottleneck=bottleneck)
    else:
        model = TransformerRegressor(len(itos), cfg["EMBED_DIM"], ydim, pad_idx,
                                     nheads=cfg.get("NHEADS",4), nlayers=cfg.get("NLAYERS",2),
                                     ffn_dim=cfg.get("FFN_DIM",512), pe_dropout=cfg.get("PE_DROPOUT",0.1),
                                     head_dropout=cfg.get("DROPOUT",0.0), pool=pool, bottleneck=bottleneck)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(DEVICE).eval()

    rows = list(read_jsonl(Path(args.data)))
    key32, key512 = "assistant_y32", "assistant_y512"
    has32 = sum(1 for r in rows if isinstance(r.get(key32), list) and len(r[key32])==32)
    has512 = sum(1 for r in rows if isinstance(r.get(key512), list) and len(r[key512])==512)
    both = sum(1 for r in rows if isinstance(r.get(key32), list) and isinstance(r.get(key512), list))
    print(f"Data counts → y32: {has32}  y512: {has512}  both: {both}  | ckpt expects: {ydim}")

    target_key = key32 if ydim == 32 else key512
    pairs = [(get_user_text(o), torch.tensor(o[target_key], dtype=torch.float32))
             for o in rows if isinstance(o.get(target_key), list) and len(o[target_key])==ydim]

    if not pairs:
        print("No pairs matching the checkpoint's target dimension."); return

    random.seed(args.seed)
    idx = list(range(len(pairs))); random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_split))
    val_idx = set(idx[:n_val])
    if args.split == "val":
        use = [pairs[i] for i in sorted(val_idx)]
    elif args.split == "train":
        use = [pairs[i] for i in sorted(set(range(len(pairs))) - val_idx)]
    else:
        use = [pairs[i] for i in range(len(pairs))]

    empty_idx = stoi.get("<empty>"); cls_idx = stoi.get("<cls>"); unk_idx = stoi.get("<unk>",1)
    Xs, Ls, Ys = [], [], []
    for (txt, y) in use:
        toks = tokenize(txt)
        if not toks and empty_idx is not None: toks = ["<empty>"]
        ids = [stoi.get(t, unk_idx) for t in toks[:max_tokens]]
        if cls_idx is not None: ids = [cls_idx] + ids
        if not ids: ids = [unk_idx]
        x = torch.tensor(ids, dtype=torch.long)
        Xs.append(x); Ls.append(x.numel()); Ys.append(y)
    B = len(Xs); T = max(Ls)
    Xpad = torch.full((B, T), pad_idx, dtype=torch.long)
    for i, x in enumerate(Xs): Xpad[i, :x.numel()] = x
    Lten = torch.tensor(Ls, dtype=torch.long)
    Ymat = torch.stack(Ys)

    with torch.no_grad():
        pred = model(Xpad.to(DEVICE), Lten.to(DEVICE)).cpu()

    def unit_norm(t):
        return t / (t.norm(dim=-1, keepdim=True) + 1e-8)

    # normalized evaluation if requested
    if args.eval_norm:
        pred_n = unit_norm(pred); y_n = unit_norm(Ymat)
        mse_vals = ((pred_n - y_n)**2).mean(dim=1).tolist()
        cos_vals = (pred_n * y_n).sum(dim=-1).tolist()
        tag = "[Normalized space]"
    else:
        mse_vals = ((pred - Ymat)**2).mean(dim=1).tolist()
        pred_n = unit_norm(pred); y_n = unit_norm(Ymat)
        cos_vals = (pred_n * y_n).sum(dim=-1).tolist()
        tag = None

    print(f"Examples: {len(cos_vals)}")
    print(f"MSE: {mean(mse_vals):.6f}")
    mc = mean(cos_vals); md = median(cos_vals)
    print(f"Cosine: mean={mc:.4f}  median={md:.4f}  (mean angle ≈ {math.degrees(math.acos(max(-1.0,min(1.0,mc)))):.1f}°)")
    pct = lambda th: 100.0 * sum(c >= th for c in cos_vals) / max(1, len(cos_vals))
    print(f"Thresholds:  ≥0.80={pct(0.80):.1f}%   ≥0.90={pct(0.90):.1f}%   ≥0.95={pct(0.95):.1f}%")
    if tag:
        print(f"\n{tag}")
        nmse = ((unit_norm(pred)-unit_norm(Ymat))**2).mean(dim=1).tolist()
        print(f"Norm-MSE: {mean(nmse):.6f}")
        print(f"Cosine: mean={mc:.4f}  median={md:.4f}  (mean angle ≈ {math.degrees(math.acos(max(-1.0,min(1.0,mc)))):.1f}°)")
        print(f"Thresholds:  ≥0.80={pct(0.80):.1f}%   ≥0.90={pct(0.90):.1f}%   ≥0.95={pct(0.95):.1f}%")

    # Buckets by input length (excluding <cls>)
    eff_len = (Lten - 1).clamp(min=1).tolist()
    buckets = [(0,8,"≤8"), (9,32,"9–32"), (33,128,"33–128"), (129,10**9,"129+")]
    print("\n[By input length (non-pad tokens)]")
    for lo, hi, name in buckets:
        idxs = [i for i,n in enumerate(eff_len) if lo <= n <= hi]
        if not idxs:
            print(f"{name:>7}: n={0:3d}"); continue
        cos_b = [cos_vals[i] for i in idxs]
        m = mean(cos_b)
        print(f"{name:>7}: n={len(idxs):3d}  cos={m:.4f}  ≥0.90={100.0*sum(c>=0.9 for c in cos_b)/len(cos_b):.1f}%"
              f"  | norm_cos={m:.4f}  ≥0.90={100.0*sum(c>=0.9 for c in cos_b)/len(cos_b):.1f}%")

if __name__ == "__main__":
    main()

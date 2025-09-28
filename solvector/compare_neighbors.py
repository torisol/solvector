# compare_neighbors.py
import argparse, json, math, re
from pathlib import Path
import torch, torch.nn as nn

# ---------------------------
# Tokenization
# ---------------------------
TOKEN_RE = re.compile(r"\w+|\S")
def tokenize(t): return [x.lower() for x in TOKEN_RE.findall(t or "")]

# ---------------------------
# Small NN blocks
# ---------------------------
def make_head(in_dim, y_dim, dropout=0.0, bottleneck=0):
    if bottleneck and bottleneck > 0:
        return nn.Sequential(
            nn.Linear(in_dim, bottleneck), nn.ReLU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(bottleneck, y_dim)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(in_dim, y_dim)
        )

class GRURegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, y_dim, pad_idx,
                 dropout=0.0, bottleneck=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = make_head(hidden_dim, y_dim, dropout=dropout, bottleneck=bottleneck)
    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout); self.batch_first=batch_first
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
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
                 nheads=4, nlayers=2, ffn_dim=512, pe_dropout=0.1, head_dropout=0.0,
                 pool="mean", bottleneck=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pe  = PositionalEncoding(embed_dim, dropout=pe_dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads,
                                           dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = make_head(embed_dim, y_dim, dropout=head_dropout, bottleneck=bottleneck)
        self.pad_idx = pad_idx; self.pool = pool
        if pool == "attn": self.attnpool = AttnPool(embed_dim)
    def forward(self, x, lengths):
        mask = (x == self.pad_idx)
        h = self.emb(x); h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        if self.pool == "cls":
            z = h[:, 0, :]
        elif self.pool == "mean":
            valid = (x != self.pad_idx)
            if valid.size(1) > 0: valid[:, 0] = False  # exclude <cls>
            denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1).float()
            z = (h.masked_fill((~valid).unsqueeze(-1), 0.0).sum(dim=1)) / denom
        else:
            z = self.attnpool(h, mask)
        return self.head(z)

# ---------------------------
# Helpers
# ---------------------------
def encode(text, stoi, max_tokens, unk_idx=1, empty_idx=None, cls_idx=None, use_cls=True):
    toks = tokenize(text)
    if not toks and empty_idx is not None: toks = ["<empty>"]
    ids = [stoi.get(t, unk_idx) for t in toks[:max_tokens]]
    if use_cls and (cls_idx is not None): ids = [cls_idx] + ids
    if not ids: ids = [unk_idx]
    X = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    L = torch.tensor([X.size(1)], dtype=torch.long)
    return X, L

def cosine(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def get_assistant_text(obj):
    if "assistant_text" in obj and isinstance(obj["assistant_text"], str):
        return obj["assistant_text"]
    a = obj.get("assistant") or {}
    if isinstance(a, dict): return a.get("text") or a.get("content") or ""
    return obj.get("assistant") if isinstance(obj.get("assistant"), str) else ""

def style_summary(texts):
    toks = " ".join((t or "") for t in texts).lower()
    bullets = []
    if re.search(r"\bthanks\b|\bappreciate\b|\bglad\b", toks):
        bullets.append("warm tone; acknowledge once")
    if re.search(r"\bwould you like\b|\bwant me to\b|\bpreference\b|\boption\b", toks):
        bullets.append("ask a quick preference question before diving")
    if re.search(r"\bsteps\b|\b1\)|\b2\)|•|- ", toks):
        bullets.append("use concise structure: 1 short paragraph + ≤2 bullets")
    if re.search(r"```|code\b", toks):
        bullets.append("avoid code unless explicitly requested")
    json_like = 0
    json_re = re.compile(r"\{[^{}]{0,400}\}|\bjson\b|\b\"prompt\"\s*:")
    for t in texts:
        if json_re.search((t or "").lower()): json_like += 1
    if json_like >= 2:
        bullets.append("offer tool/JSON path only as an option")
    if re.search(r"\bsensitive\b|\bavoid\b|\bcareful\b", toks):
        bullets.append("be considerate with potentially sensitive content")
    if len(bullets) < 3:
        bullets.append("be concise and friendly")
    out, seen = [], set()
    for b in bullets:
        if b not in seen: out.append(b); seen.add(b)
    return out[:6]

# ---------------------------
# Core routine
# ---------------------------
def topk_for_ckpt(ckpt_path, data_path, text, topk=8, max_tokens_override=0, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    blob = torch.load(Path(ckpt_path), map_location=device)
    cfg, stoi, itos = blob["config"], blob["stoi"], blob["itos"]

    pad_idx = cfg["PAD_IDX"]; ydim = cfg["Y_DIM"]
    enc = cfg.get("ENCODER","gru"); pool = cfg.get("POOL","mean")
    bottleneck = cfg.get("BOTTLENECK", 0)
    max_tokens = max_tokens_override if max_tokens_override > 0 else cfg["MAX_TOKENS"]
    empty_idx = stoi.get("<empty>"); cls_idx = stoi.get("<cls>")

    # Build model with SAME head shape as ckpt (bottleneck-aware)
    if enc == "gru":
        model = GRURegressor(len(itos), cfg["EMBED_DIM"], cfg["HIDDEN_DIM"], ydim, pad_idx,
                             dropout=cfg.get("DROPOUT",0.0), bottleneck=bottleneck)
    else:
        model = TransformerRegressor(len(itos), cfg["EMBED_DIM"], ydim, pad_idx,
                                     nheads=cfg.get("NHEADS",4), nlayers=cfg.get("NLAYERS",2),
                                     ffn_dim=cfg.get("FFN_DIM",512), pe_dropout=cfg.get("PE_DROPOUT",0.1),
                                     head_dropout=cfg.get("DROPOUT",0.0), pool=pool,
                                     bottleneck=bottleneck)
    # Relaxed load to tolerate future changes (e.g., attn pool params)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(device).eval()

    # Predict yhat
    X, L = encode(text, stoi, max_tokens, unk_idx=stoi.get("<unk>",1),
                  empty_idx=empty_idx, cls_idx=cls_idx, use_cls=True)
    with torch.no_grad():
        yhat = model(X.to(device), L.to(device)).squeeze(0)
    yhat_n = yhat / (yhat.norm() + 1e-8)

    # Candidates
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    key = "assistant_y32" if ydim == 32 else "assistant_y512"

    cands = []
    for i, obj in enumerate(rows):
        y = obj.get(key)
        if isinstance(y, list) and len(y) == ydim:
            txt = get_assistant_text(obj)
            cands.append((i, torch.tensor(y, dtype=torch.float32), txt or ""))

    # Cosine ranking
    scores = []
    with torch.no_grad():
        for i, y, txt in cands:
            y_n = y.to(device) / (y.to(device).norm() + 1e-8)
            c = float((yhat_n * y_n).sum().item())
            scores.append((c, i, txt))
    scores.sort(reverse=True)
    out = [{"rank": r+1, "cos": float(c), "index": int(i), "assistant_text": txt}
           for r,(c,i,txt) in enumerate(scores[:topk])]
    return out, style_summary([o["assistant_text"] for o in out]), {
        "ENCODER": enc, "POOL": pool, "Y_DIM": ydim, "MAX_TOKENS": max_tokens, "BOTTLENECK": bottleneck
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare neighbors from two checkpoints for one prompt")
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="Override inference token budget (default: use ckpt config)")
    args = ap.parse_args()

    A, Astyle, Acfg = topk_for_ckpt(args.ckpt_a, args.data, args.text, args.topk, args.max_tokens)
    B, Bstyle, Bcfg = topk_for_ckpt(args.ckpt_b, args.data, args.text, args.topk, args.max_tokens)

    setA, setB = set(r["index"] for r in A), set(r["index"] for r in B)
    overlap = len(setA & setB) * 100.0 / max(1, len(setA | setB))
    avgA = sum(r["cos"] for r in A) / max(1, len(A))
    avgB = sum(r["cos"] for r in B) / max(1, len(B))

    print(f"[A] ENC={Acfg['ENCODER']} POOL={Acfg['POOL']} Y={Acfg['Y_DIM']} BOTTLENECK={Acfg['BOTTLENECK']} MAXTOK={Acfg['MAX_TOKENS']}  avg_cos={avgA:.4f}")
    print(f"[B] ENC={Bcfg['ENCODER']} POOL={Bcfg['POOL']} Y={Bcfg['Y_DIM']} BOTTLENECK={Bcfg['BOTTLENECK']} MAXTOK={Bcfg['MAX_TOKENS']}  avg_cos={avgB:.4f}")
    print(f"Overlap@{args.topk}: {overlap:.1f}%\n")

    def show(tag, rows):
        print(f"[{tag}] top{len(rows)}")
        for r in rows:
            snippet = (r["assistant_text"] or "").replace("\n"," ")
            if len(snippet) > 140: snippet = snippet[:137] + "..."
            print(f" #{r['rank']:>2} cos={r['cos']:.4f} idx={r['index']}  {snippet}")
        print()

    show("A", A)
    show("B", B)

    print("[style A]")
    for b in Astyle: print(f"- {b}")
    print("\n[style B]")
    for b in Bstyle: print(f"- {b}")

    difA = [b for b in Astyle if b not in Bstyle]
    difB = [b for b in Bstyle if b not in Astyle]
    if difA or difB:
        print("\n[style Δ]")
        if difA: print(" only-in-A:", "; ".join(difA))
        if difB: print(" only-in-B:", "; ".join(difB))

if __name__ == "__main__":
    main()

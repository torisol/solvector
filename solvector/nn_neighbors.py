# nn_neighbors.py
import argparse, json, math, re
from pathlib import Path
import torch, torch.nn as nn

# --- tokenization ---
TOKEN_RE = re.compile(r"\w+|\S")
def tokenize(t): return [x.lower() for x in TOKEN_RE.findall(t or "")]

# --- small models (match train/eval/predict) ---
class GRURegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, y_dim, pad_idx, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(hidden_dim, y_dim)
        )
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
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        if self.batch_first: x = x + self.pe[:, :x.size(1), :]
        else:                 x = x + self.pe[:, :x.size(0), :].transpose(0,1)
        return self.dropout(x)

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
    def forward(self, h, mask):
        # h: (B,T,E), mask: (B,T) True at pads
        q = self.q.view(1,1,-1)                 # (1,1,E)
        scores = (h * q).sum(dim=-1)            # (B,T)
        scores = scores.masked_fill(mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        return (w * h).sum(dim=1)               # (B,E)

class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, y_dim, pad_idx,
                 nheads=4, nlayers=2, ffn_dim=512,
                 pe_dropout=0.1, head_dropout=0.0, pool="mean"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pe  = PositionalEncoding(embed_dim, dropout=pe_dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads,
                                           dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(head_dropout) if head_dropout>0 else nn.Identity(),
            nn.Linear(embed_dim, y_dim)
        )
        self.pad_idx = pad_idx
        self.pool = pool
        if pool == "attn":
            self.attnpool = AttnPool(embed_dim)

    def forward(self, x, lengths):
        mask = (x == self.pad_idx)  # (B,T)
        h = self.emb(x); h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        if self.pool == "cls":
            z = h[:, 0, :]
        elif self.pool == "mean":
            valid = (x != self.pad_idx)
            if valid.size(1) > 0: valid[:, 0] = False  # exclude <cls>
            denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1).float()
            z = (h.masked_fill((~valid).unsqueeze(-1), 0.0).sum(dim=1)) / denom
        else:  # 'attn'
            z = self.attnpool(h, mask)
        return self.head(z)

# --- helpers ---
def cosine(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def encode(text, stoi, max_tokens, unk_idx=1, empty_idx=None, cls_idx=None, use_cls=True):
    toks = tokenize(text)
    if not toks and empty_idx is not None:
        toks = ["<empty>"]
    ids = [stoi.get(t, unk_idx) for t in toks[:max_tokens]]
    if use_cls and (cls_idx is not None):
        ids = [cls_idx] + ids
    if not ids:
        ids = [unk_idx]
    X = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    L = torch.tensor([X.size(1)], dtype=torch.long)
    return X, L

def get_assistant_text(obj):
    if "assistant_text" in obj and isinstance(obj["assistant_text"], str):
        return obj["assistant_text"]
    a = obj.get("assistant") or {}
    if isinstance(a, dict):
        return a.get("text") or a.get("content") or ""
    return obj.get("assistant") if isinstance(obj.get("assistant"), str) else ""

# --- simple heuristic style summarizer ---
def style_summary(neighbor_texts):
    import re
    texts = [(t or "") for t in neighbor_texts]
    toks = " ".join(texts).lower()

    bullets = []

    # tone / acknowledge
    if re.search(r"\bthanks\b|\bappreciate\b|\bglad\b", toks):
        bullets.append("warm tone; acknowledge once")

    # preference question
    if re.search(r"\bwould you like\b|\bwant me to\b|\bpreference\b|\boption\b", toks):
        bullets.append("ask a quick preference question before diving")

    # structure
    if re.search(r"\bsteps\b|\b1\)|\b2\)|•|- ", toks):
        bullets.append("use concise structure: 1 short paragraph + ≤2 bullets")

    # code/tooling
    if re.search(r"```|code\b", toks):
        bullets.append("avoid code unless explicitly requested")

    # JSON downweighting: include only if ≥2 neighbors show JSON-ish patterns
    json_like = 0
    json_re = re.compile(r"\{[^{}]{0,400}\}|\bjson\b|\b\"prompt\"\s*:")  # cheap heuristic
    for txt in texts:
        if json_re.search(txt.lower()):
            json_like += 1
    if json_like >= 2:
        bullets.append("offer tool/JSON path only as an option")

    # sensitivity
    if re.search(r"\bsensitive\b|\bavoid\b|\bcareful\b", toks):
        bullets.append("be considerate with potentially sensitive content")

    # fallback
    if len(bullets) < 3:
        bullets.append("be concise and friendly")

    # dedupe & cap
    seen, out = set(), []
    for b in bullets:
        if b not in seen:
            seen.add(b); out.append(b)
    return out[:6]

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Nearest neighbors for predicted y-vector (plus style card)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="JSONL with assistant_y32/y512")
    ap.add_argument("--text", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mincos", type=float, default=-1.0)
    ap.add_argument("--write", type=str, default="", help="Optional output JSONL of neighbors")
    ap.add_argument("--style-out", type=str, default="out/style_card.txt", help="Path to write style card")
    ap.add_argument("--max-tokens", type=int, default=0,
                help="Override max tokens at inference (default: use ckpt config)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(args.ckpt), map_location=device)
    cfg, stoi, itos = blob["config"], blob["stoi"], blob["itos"]
    pad_idx = cfg["PAD_IDX"]; max_tokens = cfg["MAX_TOKENS"]
    ydim = cfg["Y_DIM"]; enc = cfg.get("ENCODER","gru"); pool = cfg.get("POOL","mean")
    empty_idx = stoi.get("<empty>"); cls_idx = stoi.get("<cls>")
    max_tokens_cfg = cfg["MAX_TOKENS"]
    max_tokens = args.max_tokens if args.max_tokens > 0 else max_tokens_cfg


    # model
    if enc == "gru":
        model = GRURegressor(len(itos), cfg["EMBED_DIM"], cfg["HIDDEN_DIM"], ydim, pad_idx, dropout=cfg.get("DROPOUT",0.0))
    else:
        model = TransformerRegressor(
            vocab_size=len(itos), embed_dim=cfg["EMBED_DIM"], y_dim=ydim, pad_idx=pad_idx,
            nheads=cfg.get("NHEADS",4), nlayers=cfg.get("NLAYERS",2), ffn_dim=cfg.get("FFN_DIM",512),
            pe_dropout=cfg.get("PE_DROPOUT",0.1), head_dropout=cfg.get("DROPOUT",0.0), pool=pool
        )
    # relaxed load to tolerate extra keys (e.g., attn params)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(device).eval()
    print(f"[ckpt] ENCODER={enc}  POOL={pool if enc!='gru' else '-'}  Y_DIM={ydim}")

    # predict yhat
    X, L = encode(args.text, stoi, max_tokens, unk_idx=stoi.get("<unk>",1),
                empty_idx=empty_idx, cls_idx=cls_idx, use_cls=True)

    with torch.no_grad():
        yhat = model(X.to(device), L.to(device)).squeeze(0)
    yhat_n = yhat / (yhat.norm() + 1e-8)

    # load dataset vectors/texts
    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
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

    # cosine to all candidates
    scores = []
    with torch.no_grad():
        for i, y, txt in cands:
            y_n = y.to(device) / (y.to(device).norm() + 1e-8)
            c = float((yhat_n * y_n).sum().item())
            if c >= args.mincos:
                scores.append((c, i, txt))

    scores.sort(reverse=True)
    K = min(args.topk, len(scores))
    print(f"\n[neighbors] top {K} (of {len(scores)}) by cosine for: {args.text!r}\n")
    out_rows = []
    for rank, (c, idx, txt) in enumerate(scores[:K], 1):
        snippet = (txt or "").replace("\n"," ")
        if len(snippet) > 160: snippet = snippet[:157] + "..."
        print(f" #{rank:>2}  cos={c:.4f}  idx={idx}  text≈ {snippet}")
        out_rows.append({"rank": rank, "cos": c, "index": idx, "assistant_text": txt})

    # write neighbors (optional)
    if args.write:
        outp = Path(args.write); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(out_rows)} neighbors → {args.write}")

    # style card
    style_lines = style_summary([r["assistant_text"] for r in out_rows])
    print("\n[style]")
    for b in style_lines: print(f"- {b}")

    # write style card
    style_path = Path(args.style_out)
    style_path.parent.mkdir(parents=True, exist_ok=True)
    with style_path.open("w", encoding="utf-8") as f:
        for b in style_lines:
            f.write(f"- {b}\n")
    print(f"\nWrote style card → {style_path}")

if __name__ == "__main__":
    main()

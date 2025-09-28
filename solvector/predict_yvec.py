import argparse, torch
from pathlib import Path
import math, re, json
from pathlib import Path
import torch, torch.nn as nn

TOKEN_RE = re.compile(r"\w+|\S")
def tokenize(t): return [x.lower() for x in TOKEN_RE.findall(t or "")]

def make_head(in_dim, y_dim, dropout=0.0, bottleneck=0):
    if bottleneck and bottleneck > 0:
        return nn.Sequential(
            nn.Linear(in_dim, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, y_dim)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
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

def build_model_from_cfg(cfg, stoi, itos):
    pad_idx = int(cfg["PAD_IDX"])
    ydim = int(cfg["Y_DIM"])
    enc = cfg.get("ENCODER","gru")
    pool = cfg.get("POOL","mean")
    bottleneck = int(cfg.get("BOTTLENECK", 0) or 0)
    if enc == "gru":
        return GRURegressor(len(itos), int(cfg["EMBED_DIM"]), int(cfg["HIDDEN_DIM"]), ydim, pad_idx,
                            dropout=float(cfg.get("DROPOUT",0.0)), bottleneck=bottleneck)
    else:
        return TransformerRegressor(len(itos), int(cfg["EMBED_DIM"]), ydim, pad_idx,
                                    nheads=int(cfg.get("NHEADS",4)), nlayers=int(cfg.get("NLAYERS",2)),
                                    ffn_dim=int(cfg.get("FFN_DIM",512)), pe_dropout=float(cfg.get("PE_DROPOUT",0.1)),
                                    head_dropout=float(cfg.get("DROPOUT",0.0)), pool=pool, bottleneck=bottleneck)


def encode(text, stoi, max_tokens, unk_idx=1, empty_idx=None, cls_idx=None, use_cls=True):
    toks = tokenize(text)
    if not toks and empty_idx is not None: toks = ["<empty>"]
    ids = [stoi.get(t, unk_idx) for t in toks[:max_tokens]]
    if use_cls and (cls_idx is not None): ids = [cls_idx] + ids
    if not ids: ids = [unk_idx]
    X = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    L = torch.tensor([X.size(1)], dtype=torch.long)
    return X, L

def main():
    ap = argparse.ArgumentParser(description="Predict y-vector for a single input")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-tokens", type=int, default=0, help="Override inference token budget")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(args.ckpt), map_location=device)
    cfg, stoi, itos = blob["config"], blob["stoi"], blob["itos"]
    model = build_model_from_cfg(cfg, stoi, itos)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(device).eval()

    max_tokens = args.max_tokens if args.max_tokens>0 else int(cfg.get("MAX_TOKENS", 256))
    X, L = encode(args.text, stoi, max_tokens, unk_idx=stoi.get("<unk>",1),
                  empty_idx=stoi.get("<empty>"), cls_idx=stoi.get("<cls>"), use_cls=True)
    with torch.no_grad():
        y = model(X.to(device), L.to(device)).squeeze(0)

    # print as space-separated floats
    print(" ".join(f"{v:.6f}" for v in y.tolist()))

if __name__ == "__main__":
    main()

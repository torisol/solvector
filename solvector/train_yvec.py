# solvector/train_yvec.py
import argparse, json, os, random, re, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TOKEN_RE = re.compile(r"\w+|\S")
def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_pair(obj, y_key: str):
    # user text
    if "user_text" in obj and isinstance(obj["user_text"], str):
        ux = obj["user_text"]
    elif "user" in obj and isinstance(obj["user"], dict):
        ux = obj["user"].get("text") or obj["user"].get("content") or ""
    elif isinstance(obj.get("user"), str):
        ux = obj["user"]
    else:
        ux = ""

    # y vector
    if y_key == "auto":
        if "assistant_y512" in obj:
            y = obj["assistant_y512"]; found_key = "assistant_y512"
        elif "assistant_y32" in obj:
            y = obj["assistant_y32"]; found_key = "assistant_y32"
        else:
            return None
    else:
        if y_key not in obj:
            return None
        y = obj[y_key]; found_key = y_key

    if not (isinstance(y, list) and all(isinstance(v,(int,float)) for v in y)):
        return None
    return ux, y, found_key

SPECIALS = ["<pad>", "<unk>", "<empty>", "<cls>"]
def build_vocab(pairs, vocab_size):
    from collections import Counter
    counter = Counter()
    for ux, _ in pairs:
        toks = tokenize(ux)
        if not toks: toks = ["<empty>"]
        counter.update(toks)
    itos = SPECIALS[:]
    remaining = max(0, vocab_size - len(itos))
    itos += [tok for tok, _ in counter.most_common(remaining) if tok not in itos]
    stoi = {t:i for i,t in enumerate(itos)}
    return itos, stoi

class YVecDataset(Dataset):
    def __init__(self, pairs, stoi, max_tokens, pad_idx, unk_idx, empty_idx, cls_idx, use_cls=True):
        self.items = pairs; self.stoi=stoi; self.max_tokens=max_tokens
        self.pad_idx=pad_idx; self.unk_idx=unk_idx; self.empty_idx=empty_idx; self.cls_idx=cls_idx
        self.use_cls=use_cls
    def __len__(self): return len(self.items)
    def encode(self, text):
        toks = tokenize(text)
        if not toks: toks = ["<empty>"]
        ids = [self.stoi.get(t, self.unk_idx) for t in toks[:self.max_tokens]]
        if self.use_cls and (self.cls_idx is not None):
            ids = [self.cls_idx] + ids
        if not ids: ids = [self.unk_idx]
        return torch.tensor(ids, dtype=torch.long)
    def __getitem__(self, i):
        ux, y = self.items[i]
        return self.encode(ux), torch.tensor(y, dtype=torch.float32)

def collate(batch, pad_idx):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    T = max(lengths)
    X = torch.full((len(xs), T), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    Y = torch.stack(ys)
    return X, Y, torch.tensor(lengths, dtype=torch.long)

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
                 nheads=4, nlayers=2, ffn_dim=512, pe_dropout=0.1, head_dropout=0.0,
                 pool="mean", bottleneck=0):
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

def cosine_sim(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def main():
    ap = argparse.ArgumentParser(description="Train user->assistant vector regressor")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--y-key", type=str, default="auto", choices=["auto","assistant_y32","assistant_y512"])
    ap.add_argument("--y-dim", type=int, default=32)

    ap.add_argument("--encoder", type=str, default="gru", choices=["gru","transformer"])
    ap.add_argument("--pool", type=str, default="mean", choices=["cls","mean","attn"])

    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--ffn-dim", type=int, default=512)
    ap.add_argument("--pe-dropout", type=float, default=0.1)
    ap.add_argument("--bottleneck", type=int, default=0)

    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="models/yvec.pt")
    ap.add_argument("--warmstart", type=str, default="")
    ap.add_argument("--freeze-epochs", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--loss", type=str, default="mse", choices=["mse","cos","mixed"])
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--contrastive", action="store_true")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--gamma", type=float, default=0.5)

    ap.add_argument("--len-weights", type=str, default="")
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--norm-targets", action="store_true")
    ap.add_argument("--norm-pred", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])

    ap.add_argument("--select", type=str, default="mse", choices=["mse","cos","mixed"])

    # NEW: reuse vocab from an existing checkpoint (needed for y512 warm-start)
    ap.add_argument("--vocab-from", type=str, default="",
                    help="Load stoi/itos/PAD/UNK/CLS from an existing checkpoint")

    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # length weights
    len_w = [1.0, 1.0, 1.0, 1.0]
    if args.len_weights:
        parts = [float(x) for x in args.len_weights.split(",")]
        for i, v in enumerate(parts[:4]): len_w[i] = v

    raw = list(read_jsonl(Path(args.data)))

    pairs = []
    y_key_used = None
    for row in raw:
        got = extract_pair(row, args.y_key)
        if got is None: continue
        ux, y, found_key = got
        if len(y) != args.y_dim: continue
        if y_key_used is None: y_key_used = found_key
        pairs.append((ux, y))

    if args.y_dim not in (32, 512):
        raise ValueError("--y-dim must be 32 or 512.")
    if len(pairs) == 0:
        raise ValueError("No training pairs found for the specified y-dim/key.")

    # Vocab: either reuse from ckpt or build fresh
    if args.vocab_from:
        vb = torch.load(Path(args.vocab_from), map_location="cpu")
        itos = vb["itos"]; stoi = vb["stoi"]
        PAD_IDX = vb["config"]["PAD_IDX"]
        UNK_IDX = stoi.get("<unk>", 1)
        EMPTY_IDX = stoi.get("<empty>", stoi.get("<pad>", 0))
        CLS_IDX = stoi.get("<cls>", None)
        print(f"[vocab] loaded from {args.vocab_from} (|V|={len(itos)})")
    else:
        itos, stoi = build_vocab(pairs, args.vocab_size)
        PAD_IDX = stoi["<pad>"]; UNK_IDX = stoi["<unk>"]
        EMPTY_IDX = stoi["<empty>"]; CLS_IDX = stoi["<cls>"]

    train_pairs = pairs[int(len(pairs)*args.val_split):]
    val_pairs   = pairs[:int(len(pairs)*args.val_split)]

    train_ds = YVecDataset(train_pairs, stoi, args.max_tokens, PAD_IDX, UNK_IDX, EMPTY_IDX, CLS_IDX, use_cls=True)
    val_ds   = YVecDataset(val_pairs,   stoi, args.max_tokens, PAD_IDX, UNK_IDX, EMPTY_IDX, CLS_IDX, use_cls=True)

    collate_fn = lambda b: collate(b, PAD_IDX)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn)

    # Model
    if args.encoder == "gru":
        model = GRURegressor(len(itos), args.embed_dim, args.hidden_dim, args.y_dim, PAD_IDX,
                             dropout=args.dropout, bottleneck=args.bottleneck).to(DEVICE)
    else:
        model = TransformerRegressor(len(itos), args.embed_dim, args.y_dim, PAD_IDX,
                                     nheads=args.nheads, nlayers=args.nlayers,
                                     ffn_dim=args.ffn_dim, pe_dropout=args.pe_dropout,
                                     head_dropout=args.dropout, pool=args.pool,
                                     bottleneck=args.bottleneck).to(DEVICE)

    # Warm-start encoder
    freeze_encoder_epochs = 0
    if args.warmstart:
        ckpt = torch.load(args.warmstart, map_location=DEVICE)
        sd = ckpt["state_dict"]; own = model.state_dict()
        enc_prefixes = ["emb.", "gru.", "encoder.", "pe.", "attnpool."]
        loaded = 0
        for k, v in sd.items():
            if any(k.startswith(p) for p in enc_prefixes):
                if k in own and own[k].shape == v.shape:
                    own[k] = v; loaded += 1
        model.load_state_dict(own, strict=False)
        freeze_encoder_epochs = max(0, args.freeze_epochs)
        print(f"Warm-started encoder from {args.warmstart} (loaded {loaded} tensors). Freeze {freeze_encoder_epochs} epochs.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def maybe_norm(t, on):
        return t / (t.norm(dim=-1, keepdim=True) + args.eps) if on else t

    def step_eval(dl):
        model.eval()
        mse_sum, cos_all, n = 0.0, [], 0
        with torch.no_grad():
            for X, Y, L in dl:
                X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
                P = model(X, L)
                Pn, Yn = maybe_norm(P, args.norm_pred), maybe_norm(Y, args.norm_targets)
                mse_sum += ((Pn - Yn)**2).mean().item() * X.size(0)
                cos_all.extend(cosine_sim(Pn, Yn).tolist())
                n += X.size(0)
        return mse_sum/max(1,n), sum(cos_all)/max(1,len(cos_all))

    best_sel = None
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Training
    for epoch in range(1, args.epochs+1):
        model.train()
        # conditionally freeze
        for n_, p in model.named_parameters():
            if epoch <= freeze_encoder_epochs and n_.startswith(("emb.","gru.","encoder.","pe.","attnpool.")):
                p.requires_grad = False
            else:
                p.requires_grad = True

        for X, Y, L in train_dl:
            X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
            P = model(X, L)
            Pn, Yn = maybe_norm(P, args.norm_pred), maybe_norm(Y, args.norm_targets)
            # base loss
            if args.loss == "mse":
                base = ((Pn - Yn)**2).mean(dim=1)
            elif args.loss == "cos":
                base = 1 - cosine_sim(Pn, Yn)
            else:
                base = ((Pn - Yn)**2).mean(dim=1) + args.alpha * (1 - cosine_sim(Pn, Yn))

            # length weights
            with torch.no_grad():
                eff = (L - 1).clamp(min=1)
                lw = torch.ones_like(eff, dtype=torch.float, device=eff.device)
                for i in range(eff.size(0)):
                    n_tok = int(eff[i].item())
                    if n_tok <= 8: lw[i] = len_w[0]
                    elif n_tok <= 32: lw[i] = len_w[1]
                    elif n_tok <= 128: lw[i] = len_w[2]
                    else: lw[i] = len_w[3]
            loss = (lw * base).mean()

            # contrastive (InfoNCE) on normalized
            if args.contrastive:
                Pn2 = Pn / (Pn.norm(dim=-1, keepdim=True) + args.eps)
                Yn2 = Yn / (Yn.norm(dim=-1, keepdim=True) + args.eps)
                logits = (Pn2 @ Yn2.t()) / max(args.tau, 1e-6)
                target = torch.arange(logits.size(0), device=logits.device)
                ce = torch.nn.functional.cross_entropy(logits, target)
                loss = loss + args.gamma * ce

            optimizer.zero_grad(); loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_mse, val_cos = step_eval(val_dl)
        print(f"Epoch {epoch:02d}/{args.epochs}  val_MSE={val_mse:.6f}  val_cos={val_cos:.4f}")

        # selection
        sel = {"mse": -val_mse, "cos": val_cos, "mixed": -val_mse + val_cos}[args.select]
        if best_sel is None or sel > best_sel:
            best_sel = sel
            torch.save({
                "state_dict": model.state_dict(),
                "stoi": stoi, "itos": itos,
                "config": {
                    "ENCODER": args.encoder, "EMBED_DIM": args.embed_dim, "HIDDEN_DIM": args.hidden_dim,
                    "Y_DIM": args.y_dim, "PAD_IDX": stoi["<pad>"], "MAX_TOKENS": args.max_tokens,
                    "VOCAB_SIZE": len(itos), "DROPOUT": args.dropout,
                    "NHEADS": args.nheads, "NLAYERS": args.nlayers, "FFN_DIM": args.ffn_dim,
                    "PE_DROPOUT": args.pe_dropout, "POOL": args.pool, "BOTTLENECK": args.bottleneck
                },
                "meta": {
                    "y_key": y_key_used, "loss_mode": args.loss, "alpha": args.alpha,
                    "norm_targets": bool(args.norm_targets), "norm_pred": bool(args.norm_pred),
                    "contrastive": bool(args.contrastive), "tau": args.tau, "gamma": args.gamma
                }
            }, out_path)
            print(f"✓ Saved best → {out_path} (select={args.select})")

if __name__ == "__main__":
    main()

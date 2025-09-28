# tools/cos_check.py
import argparse, torch, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--eval-norm", action="store_true")
args = ap.parse_args()

from solvector.evaluate import read_jsonl, get_user_text, tokenize, TransformerRegressor, GRURegressor

blob = torch.load(Path(args.ckpt), map_location="cpu")
cfg, stoi, itos = blob["config"], blob["stoi"], blob["itos"]
pad = cfg["PAD_IDX"]; ydim = cfg["Y_DIM"]; enc = cfg.get("ENCODER","gru")
max_tokens = cfg["MAX_TOKENS"]; pool = cfg.get("POOL","mean"); bottleneck = cfg.get("BOTTLENECK",0)

if enc=="gru":
    model = GRURegressor(len(itos), cfg["EMBED_DIM"], cfg.get("HIDDEN_DIM",256), ydim, pad, bottleneck=bottleneck)
else:
    model = TransformerRegressor(len(itos), cfg["EMBED_DIM"], ydim, pad,
                                 nheads=cfg.get("NHEADS",4), nlayers=cfg.get("NLAYERS",2),
                                 ffn_dim=cfg.get("FFN_DIM",512), pool=pool, bottleneck=bottleneck)
model.load_state_dict(blob["state_dict"], strict=False); model.eval()

rows = list(read_jsonl(Path(args.data)))
key = "assistant_y32" if ydim==32 else "assistant_y512"
pairs = [(get_user_text(o), torch.tensor(o[key], dtype=torch.float32))
         for o in rows if isinstance(o.get(key), list) and len(o[key])==ydim]

def enc(txt):
    toks = tokenize(txt) or ["<empty>"]
    ids = [stoi.get(t,1) for t in toks[:max_tokens]]
    cls = stoi.get("<cls>")
    if cls is not None: ids = [cls] + ids
    x = torch.tensor(ids).unsqueeze(0); L = torch.tensor([x.size(1)])
    return x, L

with torch.no_grad():
    preds, gts = [], []
    for (txt,y) in pairs[:256]:
        x,L = enc(txt)
        p = model(x, L).squeeze(0)
        preds.append(p); gts.append(y)
    P = torch.stack(preds); Y = torch.stack(gts)
    if args.eval_norm:
        P = P / (P.norm(dim=-1, keepdim=True)+1e-8)
        Y = Y / (Y.norm(dim=-1, keepdim=True)+1e-8)
    cos = ( (P/ (P.norm(dim=-1, keepdim=True)+1e-8)) * (Y/(Y.norm(dim=-1, keepdim=True)+1e-8)) ).sum(dim=-1)
    print({"mean_cos": float(cos.mean().item()), "median_cos": float(cos.median().item())})

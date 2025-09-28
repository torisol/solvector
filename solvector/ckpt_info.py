# ckpt_info.py
import argparse, json
import torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Quick checkpoint inspector")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    blob = torch.load(Path(args.ckpt), map_location="cpu")
    cfg = blob.get("config", {})
    print(json.dumps(cfg, indent=2))

    enc = cfg.get("ENCODER", "gru")
    pool = cfg.get("POOL", "-") if enc != "gru" else "-"
    print("\n[summary]")
    print(f" ENCODER={enc}  POOL={pool}")
    print(f" Y_DIM={cfg.get('Y_DIM')}  EMBED={cfg.get('EMBED_DIM')}  HIDDEN={cfg.get('HIDDEN_DIM')}")
    if enc == "transformer":
        print(f" NLAYERS={cfg.get('NLAYERS')}  NHEADS={cfg.get('NHEADS')}  FFN={cfg.get('FFN_DIM')}")
    print(f" PAD_IDX={cfg.get('PAD_IDX')}  MAX_TOKENS={cfg.get('MAX_TOKENS')}")

if __name__ == "__main__":
    main()

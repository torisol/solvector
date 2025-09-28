# compare_ckpts.py
import argparse, json, os, torch
from pathlib import Path
from datetime import datetime

FIELDS_CFG  = ["ENCODER","POOL","Y_DIM","EMBED_DIM","HIDDEN_DIM","NHEADS","NLAYERS","FFN_DIM","DROPOUT","MAX_TOKENS","PAD_IDX","SCHED","WARMUP_STEPS"]
FIELDS_META = ["loss_mode","alpha","weight_decay","grad_clip","norm_targets","norm_pred","eps","y_key","device"]

def short(p: Path, root=None):
    try:
        return str(p.relative_to(root or Path.cwd()))
    except Exception:
        return str(p)

def get(cfg, key, default="-"):
    v = cfg.get(key, default)
    return "-" if v is None else v

def load_row(path: Path):
    blob = torch.load(path, map_location="cpu")
    cfg  = blob.get("config", {})
    meta = blob.get("meta", {})
    row = {
        "CKPT": short(path),
        "UPDATED": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        **{k: get(cfg, k) for k in FIELDS_CFG},
        **{k: get(meta, k) for k in FIELDS_META},
    }
    # Pretty encoder/pool defaults
    if row["ENCODER"] == "gru":
        row["POOL"] = "-"
    if row["POOL"] == "-":
        row["POOL"] = get(cfg, "POOL", "-")
    return row

def format_table(rows, cols):
    widths = {c: max(len(c), *(len(str(r.get(c,""))) for r in rows)) for c in cols}
    def line(sep_left="|", sep_mid="|", sep_right="|"):
        parts = [sep_left]
        for i,c in enumerate(cols):
            parts.append(f" {str(c).ljust(widths[c])} ")
            parts.append(sep_mid if i < len(cols)-1 else sep_right)
        return "".join(parts)
    lines = [line(), line(sep_left="+", sep_mid="+", sep_right="+").replace(" ", "-")]
    for r in rows:
        parts = ["|"]
        for i,c in enumerate(cols):
            parts.append(f" {str(r.get(c,'')).ljust(widths[c])} ")
            parts.append("|" if i < len(cols)-1 else "|")
        lines.append("".join(parts))
    lines.append(line(sep_left="+", sep_mid="+", sep_right="+").replace(" ", "-"))
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Compare multiple yvec checkpoints")
    ap.add_argument("--ckpt", nargs="+", required=True, help="One or more checkpoint paths")
    args = ap.parse_args()

    paths = [Path(p) for p in args.ckpt]
    rows = [load_row(p) for p in paths]

    # Primary table (model/config)
    cols_main = ["CKPT","UPDATED","ENCODER","POOL","Y_DIM","EMBED_DIM","HIDDEN_DIM","NHEADS","NLAYERS","FFN_DIM","DROPOUT","MAX_TOKENS","PAD_IDX","SCHED","WARMUP_STEPS"]
    print(format_table(rows, cols_main))
    print()

    # Training meta (loss/regularization)
    cols_meta = ["CKPT","loss_mode","alpha","weight_decay","grad_clip","norm_targets","norm_pred","eps","y_key","device"]
    print(format_table(rows, cols_meta))

if __name__ == "__main__":
    main()

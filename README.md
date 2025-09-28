# solvector (v9.3 canonical)

Predictor & trainer for y32/y512 embeddings aligned with The Sol Framework.

## Quick start (Windows, CUDA 12.4)
1. `install.bat`  → creates `.venv`, installs CUDA PyTorch, installs package (editable).
2. Train y32:
   ```powershell
   .\run_train_32.bat data\train.jsonl models\yvec_tf_mean_32.pt
   ```
3. Warm-start y512 from y32:
   ```powershell
   .\run_train_512_from32.bat data\train.jsonl models\yvec_tf_mean_32.pt models\yvec_tf_mean_512.pt
   ```
4. Evaluate:
   ```powershell
   .\run_eval_val.bat  data\train.jsonl models\yvec_tf_mean_512.pt 0.1 42
   .\run_eval_norm.bat data\train.jsonl models\yvec_tf_mean_512.pt
   ```

## Repo layout
- `pyproject.toml` — package config
- `install.bat` — venv + CUDA wheel + editable install
- `run_*.bat` — launchers
- `solvector/` — package (train/evaluate/predict + utilities)
- `tools/` — helpers

## Notes
- `.gitignore` excludes venv, caches, and model artifacts.
- Ensure your data file exists at the path you pass (e.g., `data\train.jsonl`).


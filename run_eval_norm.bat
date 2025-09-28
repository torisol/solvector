@echo off
REM Usage: run_eval_norm.bat DATA_JSONL CKPT_PATH
python -m solvector.evaluate --data %1 --ckpt %2 --split val --val-split 0.1 --seed 42 --eval-norm

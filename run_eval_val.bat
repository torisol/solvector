@echo off
REM Evaluate normalized cosine on a validation split with seed.
REM Usage: run_eval_val.bat DATA_JSONL CKPT_PATH VAL_SPLIT SEED
if "%~4"=="" (
  echo Usage: run_eval_val.bat DATA_JSONL CKPT_PATH VAL_SPLIT SEED
  exit /b 1
)
python -m solvector.evaluate ^
  --data %~1 ^
  --ckpt %~2 ^
  --split val ^
  --val-split %~3 ^
  --seed %~4 ^
  --eval-norm

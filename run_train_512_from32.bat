@echo off
REM Usage: run_train_512_from32.bat DATA_JSONL CKPT32_PATH OUT_CKPT512_PATH
python -m solvector.train_yvec --data %1 --y-key assistant_y512 --y-dim 512 ^
  --encoder transformer --pool mean --embed-dim 192 --nheads 6 --nlayers 3 --ffn-dim 768 ^
  --dropout 0.2 --pe-dropout 0.05 ^
  --loss mixed --alpha 1.5 --lr 3e-5 --epochs 40 --sched cosine --warmup-steps 800 ^
  --norm-targets --norm-pred ^
  --contrastive --tau 0.07 --gamma 0.5 ^
  --len-weights 1.5,1.2,1.0,1.0 ^
  --bottleneck 128 ^
  --max-tokens 256 ^
  --vocab-from %2 ^
  --warmstart %2 --freeze-epochs 3 ^
  --select cos ^
  --out %3

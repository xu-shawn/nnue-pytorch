#!/bin/bash
set -euo pipefail
cd /workspace/nnue
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
for option in stage1 forward256 zero grouped combined; do
 case "$option" in
  stage1) opts=();;
  forward256) opts=(NNUE_FWD_THREADS=256);;
  zero) opts=(NNUE_SKIP_ZERO=1);;
  grouped) opts=(NNUE_GROUPED=1);;
  combined) opts=(NNUE_FWD_THREADS=256 NNUE_SKIP_ZERO=1 NNUE_GROUPED=1);;
 esac
 env "${opts[@]}" torchrun --standalone --nproc-per-node=4 ddp_launcher.py /workspace/forward_experiments.py /workspace/data/*.binpack --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output "/workspace/results/mature-${option}.json" > "/workspace/results/mature-${option}.log" 2>&1
done
touch /workspace/results/MATURE_BENCH_DONE

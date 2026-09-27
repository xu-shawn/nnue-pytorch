#!/bin/bash
set -euo pipefail
cd /workspace/nnue
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
checkpoint=$(cat /workspace/results/checkpoint-path.txt)
for option in cols4 cols4tile8 bucket400 keepgrads ring tree channels32; do
    case "$option" in
        cols4) opts=(NNUE_COLSPLIT=4);;
        cols4tile8) opts=(NNUE_COLSPLIT=4 NNUE_TILE=8);;
        bucket400) opts=(NNUE_BUCKET_MB=400);;
        keepgrads) opts=(NNUE_KEEP_GRADS=1);;
        ring) opts=(NCCL_ALGO=Ring);;
        tree) opts=(NCCL_ALGO=Tree);;
        channels32) opts=(NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32);;
    esac
    env "${opts[@]}" torchrun --standalone --nproc-per-node=4 ddp_launcher.py /workspace/experiments.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --resume-from-checkpoint "$checkpoint" --output "/workspace/results/cached-${option}.json" > "/workspace/results/cached-${option}.log" 2>&1
done
touch /workspace/results/SCALING_DONE

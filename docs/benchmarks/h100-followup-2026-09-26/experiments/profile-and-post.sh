#!/bin/bash
set -uo pipefail
cd /workspace/nnue
export OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
nsys --version > /workspace/results/nsys-version.txt
nsys profile --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output=/workspace/results/nsys-baseline torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py /workspace/data/*.binpack --threads 4 --num-workers 64 --warmup 64 --steps 32 --repeats 1 --random-fen-skipping 0 --resume-from-checkpoint /workspace/reproduction/lightning_logs/version_0/checkpoints/last.ckpt --nsys --output /workspace/results/nsys-baseline.json > /workspace/results/nsys-baseline.log 2>&1
printf '%s\n' "$?" > /workspace/results/nsys-exit-code.txt
bash /workspace/scaling-post.sh

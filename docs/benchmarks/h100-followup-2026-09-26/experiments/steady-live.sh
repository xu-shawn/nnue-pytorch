#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
while ! test -f /workspace/results/ALL_GPU_WORK_DONE; do sleep 5; done
cd /workspace/nnue
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
for mode in control delivered; do
    cp "/workspace/${mode}-fused-ft-kernel.py" model/modules/feature_transformer/fused_ft_kernel.py
    opts=(--ddp-bucket-cap-mb 50)
    test "$mode" = delivered && opts=(--ddp-bucket-cap-mb 400 --grouped-l1)
    torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 4 --num-workers 64 --warmup 1024 --steps 2048 --repeats 3 --random-fen-skipping 10 "${opts[@]}" --resume-from-checkpoint /workspace/mature-init.ckpt --output "/workspace/results/mature-steady-${mode}.json" > "/workspace/results/mature-steady-${mode}.log" 2>&1
done
touch /workspace/results/STEADY_DONE

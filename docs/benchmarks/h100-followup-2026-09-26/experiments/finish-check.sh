#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
while ! test -f /workspace/results/DELIVERED_DONE; do sleep 5; done
cd /workspace/nnue
python /workspace/final-density.py > /workspace/results/final-density.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
for ranks in 1 2; do
    batch=$((32768*ranks))
    torchrun --standalone --nproc-per-node="$ranks" ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --batch-size "$batch" --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output "/workspace/results/mature-final-${ranks}gpu.json" > "/workspace/results/mature-final-${ranks}gpu.log" 2>&1
done
cd /workspace/master
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/master-mature-cached.json > /workspace/results/master-mature-cached.log 2>&1
touch /workspace/results/ALL_GPU_WORK_DONE

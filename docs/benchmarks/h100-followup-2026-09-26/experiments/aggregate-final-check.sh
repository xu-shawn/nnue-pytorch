#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
test -f /workspace/results/AGGREGATE_DELIVERED_DONE
tar -xf /workspace/aggregate-format-update.tar
python -m pytest -q tests/test_aggregated_ft.py > /workspace/results/aggregate-formatted-tests.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/aggregate-final-cached.json > /workspace/results/aggregate-final-cached.log 2>&1
nsys profile --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output=/workspace/results/nsys-mature-aggregate torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 32 --repeats 1 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --nsys --output /workspace/results/nsys-mature-aggregate.json > /workspace/results/nsys-mature-aggregate.log 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv /workspace/results/nsys-mature-aggregate.nsys-rep > /workspace/results/nsys-mature-aggregate-kernels.csv 2>/workspace/results/nsys-mature-aggregate-stats.log
touch /workspace/results/AGGREGATE_FINAL_DONE

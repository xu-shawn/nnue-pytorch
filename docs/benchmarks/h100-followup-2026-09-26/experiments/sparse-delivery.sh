#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/mature-sparse-delivered-cached.json > /workspace/results/mature-sparse-delivered-cached.log 2>&1
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 4 --num-workers 64 --warmup 1024 --steps 2048 --repeats 3 --random-fen-skipping 10 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/mature-sparse-delivered-live.json > /workspace/results/mature-sparse-delivered-live.log 2>&1
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 1 --epoch-size 33554432 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/sparse-resume --resume-from-checkpoint /workspace/delivered-resume/lightning_logs/version_0/checkpoints/last.ckpt > /workspace/results/sparse-delivered-resume.log 2>&1
touch /workspace/results/SPARSE_DELIVERED_DONE

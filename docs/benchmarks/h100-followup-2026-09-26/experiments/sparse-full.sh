#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
NNUE_SPARSE_TEST=1 python /workspace/sparse_experiments.py > /workspace/results/sparse-tests.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
run() {
 local label=$1 sparse=$2 checkpoint=$3
 NNUE_SPARSE="$sparse" torchrun --standalone --nproc-per-node=4 ddp_launcher.py /workspace/sparse_experiments.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint "$checkpoint" --output "/workspace/results/${label}.json" > "/workspace/results/${label}.log" 2>&1
}
run mature-sparse-cached 1 /workspace/mature-init.ckpt
run early-dense-cached 0 /workspace/reproduction/lightning_logs/version_0/checkpoints/last.ckpt
run early-sparse-cached 1 /workspace/reproduction/lightning_logs/version_0/checkpoints/last.ckpt
touch /workspace/results/SPARSE_FULL_DONE

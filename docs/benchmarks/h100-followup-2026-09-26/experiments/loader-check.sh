#!/bin/bash
set -euo pipefail
cd /workspace/nnue
for attempt in $(seq 1 90); do
    test -f /workspace/results/READY && break
    sleep 5
done
test -f /workspace/results/READY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
# One complete production-sized training epoch; shorten validation to 8 batches.
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 1 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/reproduction > /workspace/results/reproduction.log 2>&1
checkpoint=$(find /workspace/reproduction -name '*.ckpt' | sort | tail -1)
test -n "$checkpoint"
printf '%s\n' "$checkpoint" > /workspace/results/checkpoint-path.txt
for skip in 10 0; do
    torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping "$skip" --resume-from-checkpoint "$checkpoint" --output "/workspace/results/live-skip${skip}.json" > "/workspace/results/live-skip${skip}.log" 2>&1
done
touch /workspace/results/LOADER_DONE

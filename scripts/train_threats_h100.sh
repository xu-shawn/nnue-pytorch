#!/usr/bin/env bash
# threats.yaml at vondele/nettest@9bc8d4f4f89b9126079176dfdcd901d7bf294f7d.
# Pass complete local binpack paths, followed by any train.py overrides.
# Batch size is global: 131072 / 4 = 32768 positions per GPU.
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 DATA.binpack [MORE.binpack ...] [train.py options]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
# Sparse batches exceed glibc's usual per-thread heap size. Reusing these
# allocations avoids expensive mmap/munmap turnover on every batch.
# Existing allocator settings take precedence.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-134217728}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-268435456}"

exec torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    ddp_launcher.py train.py \
    --accelerator cuda --threads "${NNUE_TORCH_THREADS:-1}" --num-workers "${NNUE_LOADER_WORKERS:-32}" \
    --compile-backend inductor --data-loader-queue-size 4 --ddp-bucket-cap-mb 400 \
    --features 'Full_Threats+PP_3Wide+HalfKAv2_hm^' --l1 1024 --l2 32 --grouped-l1 --sparse-l1 \
    --optimizer-name adamw --lr 0.4e-3 --factorized-weight-decay 4.0e-5 \
    --batch-size 131072 --epoch-size 134217728 --max-epochs 4500 \
    --validation-size 268435456 --check-val-every-n-epoch 50 \
    --random-fen-skipping 10 --early-fen-skipping 18 --soft-early-fen-skipping 32 \
    --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
    --ply-x1 0.00 --ply-y1 0.025 --ply-x2 22.0 --ply-y2 0.05 \
    --ply-x3 25.5 --ply-y3 0.20 --ply-x4 29.5 --ply-y4 0.80 \
    --pow-exp 2.4340402395048404 --qp-asymmetry 0.22866886710086187 \
    --in-scaling 295.6539508488627 --out-scaling 379.98724077106635 \
    --in-offset 285.2706341467852 --out-offset 289.1258344152218 \
    --one-cycle-steps 5108000 --one-cycle-warmup-pct 0.05 --one-cycle-final-div 1e3 \
    --lambda-schedule-steps 5052000 --lambda-cycle-jitter \
    --jitter-lambda-sample 0.0035 --jitter-lambda-batch 0.0070 \
    --jitter-decay-lambda-batch 0.999 --start-lambda 1.0 --end-lambda 1.0 \
    --lambda-cycle-warmup-pct 0.25 --lambda-cycle-delta -0.3 \
    "$@"

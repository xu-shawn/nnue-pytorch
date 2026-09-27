#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
python -m pytest -q tests/test_aggregated_ft.py tests/test_grouped_linear.py tests/test_fused_double_ft.py tests/test_sparse_linear_device_parity.py tests/test_feature_transformer.py tests/test_loss_metrics.py tests/test_optimizers.py tests/test_ddp_loader.py tests/test_nan_abort.py tests/test_lambda_scheduler.py > /workspace/results/aggregate-delivered-tests.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/aggregate-delivered-cached.json > /workspace/results/aggregate-delivered-cached.log 2>&1
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 4 --num-workers 64 --warmup 1024 --steps 2048 --repeats 3 --random-fen-skipping 10 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/aggregate-delivered-live.json > /workspace/results/aggregate-delivered-live.log 2>&1
torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/reproduction/lightning_logs/version_0/checkpoints/last.ckpt --output /workspace/results/aggregate-delivered-early.json > /workspace/results/aggregate-delivered-early.log 2>&1
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 1 --epoch-size 33554432 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/aggregate-resume --resume-from-checkpoint /workspace/sparse-resume/lightning_logs/version_0/checkpoints/last.ckpt > /workspace/results/aggregate-delivered-resume.log 2>&1
touch /workspace/results/AGGREGATE_DELIVERED_DONE

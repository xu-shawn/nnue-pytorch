#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
while ! test -f /workspace/results/MATURE_BENCH_DONE; do sleep 5; done
tar -xf /workspace/delivery-update.tar -C /workspace/nnue
cp model/modules/feature_transformer/fused_ft_kernel.py /workspace/delivered-fused-ft-kernel.py
python -m pytest -q tests/test_grouped_linear.py tests/test_fused_double_ft.py tests/test_sparse_linear_device_parity.py tests/test_feature_transformer.py tests/test_loss_metrics.py tests/test_optimizers.py tests/test_ddp_loader.py tests/test_nan_abort.py tests/test_lambda_scheduler.py > /workspace/results/delivered-tests.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
bench() {
 local label=$1 mode=$2 kind=$3 checkpoint=$4
 shift 4
 local bucket=400
 local opts=(--grouped-l1)
 cp "/workspace/${mode}-fused-ft-kernel.py" model/modules/feature_transformer/fused_ft_kernel.py
 if test "$mode" = control; then bucket=50; opts=(); fi
 if test "$kind" = cached; then opts+=(--threads 1 --num-workers 32 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0); else opts+=(--threads 4 --num-workers 64 --random-fen-skipping 10); fi
 torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" "${opts[@]}" --ddp-bucket-cap-mb "$bucket" --warmup 64 --steps 512 --repeats 3 --resume-from-checkpoint "$checkpoint" --output "/workspace/results/${label}.json" "$@" > "/workspace/results/${label}.log" 2>&1
}
bench mature-cached-control control cached /workspace/mature-init.ckpt
bench mature-cached-delivered delivered cached /workspace/mature-init.ckpt
bench mature-live-a1 control live /workspace/mature-init.ckpt
bench mature-live-b1 delivered live /workspace/mature-init.ckpt
bench mature-live-b2 delivered live /workspace/mature-init.ckpt
bench mature-live-a2 control live /workspace/mature-init.ckpt
bench early-live-delivered delivered live /workspace/reproduction/lightning_logs/version_0/checkpoints/last.ckpt
for mode in control delivered; do
 cp "/workspace/${mode}-fused-ft-kernel.py" model/modules/feature_transformer/fused_ft_kernel.py
 opts=(--grouped-l1 --ddp-bucket-cap-mb 400)
 test "$mode" = control && opts=(--ddp-bucket-cap-mb 50)
 nsys profile --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output="/workspace/results/nsys-mature-${mode}" torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 32 --repeats 1 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 "${opts[@]}" --resume-from-checkpoint /workspace/mature-init.ckpt --nsys --output "/workspace/results/nsys-mature-${mode}.json" > "/workspace/results/nsys-mature-${mode}.log" 2>&1
 nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format csv --output "/workspace/results/nsys-mature-${mode}-stats" "/workspace/results/nsys-mature-${mode}.nsys-rep" > "/workspace/results/nsys-mature-${mode}-stats.log" 2>&1
done
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 1 --check-val-every-n-epoch 1 --default-root-dir /workspace/delivered-training --resume-from-checkpoint /workspace/mature-init.ckpt > /workspace/results/delivered-training.log 2>&1
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 1 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/delivered-resume --resume-from-checkpoint /workspace/delivered-training/lightning_logs/version_0/checkpoints/last.ckpt > /workspace/results/delivered-resume.log 2>&1
touch /workspace/results/DELIVERED_DONE

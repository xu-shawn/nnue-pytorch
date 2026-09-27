#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
while ! test -f /workspace/results/SCALING_DONE; do sleep 5; done
cp model/modules/feature_transformer/fused_ft_kernel.py /workspace/control-fused-ft-kernel.py
tar -xf /workspace/final-update.tar -C /workspace/nnue
cp model/modules/feature_transformer/fused_ft_kernel.py /workspace/final-fused-ft-kernel.py
python -m pytest -q tests/test_fused_double_ft.py tests/test_sparse_linear_device_parity.py tests/test_feature_transformer.py tests/test_loss_metrics.py tests/test_optimizers.py tests/test_ddp_loader.py tests/test_nan_abort.py tests/test_lambda_scheduler.py > /workspace/results/final-tests.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
checkpoint=$(cat /workspace/results/checkpoint-path.txt)
bench() {
 local label=$1 mode=$2 bucket=$3 skip=$4
 shift 4
 cp "/workspace/${mode}-fused-ft-kernel.py" model/modules/feature_transformer/fused_ft_kernel.py
 torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 4 --num-workers 64 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping "$skip" --ddp-bucket-cap-mb "$bucket" --resume-from-checkpoint "$checkpoint" --output "/workspace/results/${label}.json" "$@" > "/workspace/results/${label}.log" 2>&1
}
bench live-control-a1 control 50 0
bench live-final-b1 final 400 0
bench live-final-b2 final 400 0
bench live-control-a2 control 50 0
bench cached-final final 400 0 --cached 8 --batch-cache /workspace/cache-skip0 --threads 1 --num-workers 32
bench live-final-skip10 final 400 10
for mode in control final; do
 cp "/workspace/${mode}-fused-ft-kernel.py" model/modules/feature_transformer/fused_ft_kernel.py
 bucket=50
 test "$mode" = final && bucket=400
 nsys profile --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output="/workspace/results/nsys-cached-${mode}" torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 32 --repeats 1 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --ddp-bucket-cap-mb "$bucket" --resume-from-checkpoint "$checkpoint" --nsys --output "/workspace/results/nsys-cached-${mode}.json" > "/workspace/results/nsys-cached-${mode}.log" 2>&1
 nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format csv --output "/workspace/results/nsys-cached-${mode}-stats" "/workspace/results/nsys-cached-${mode}.nsys-rep" > "/workspace/results/nsys-cached-${mode}-stats.log" 2>&1
done
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 2 --random-fen-skipping 0 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/final-training --resume-from-checkpoint "$checkpoint" > /workspace/results/final-training.log 2>&1
final_checkpoint=/workspace/final-training/lightning_logs/version_0/checkpoints/last.ckpt
bash scripts/train_threats_h100.sh "${data[@]}" --max-epochs 3 --random-fen-skipping 0 --validation-size 1048576 --check-val-every-n-epoch 1 --default-root-dir /workspace/final-resume --resume-from-checkpoint "$final_checkpoint" > /workspace/results/final-resume.log 2>&1
touch /workspace/results/FINAL_DONE

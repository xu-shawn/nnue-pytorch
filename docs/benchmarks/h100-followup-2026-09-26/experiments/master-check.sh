#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
while ! test -f /workspace/results/FINAL_DONE; do sleep 5; done
ncu --set basic --kernel-name regex:fused_double_ft_backward --launch-count 1 --export /workspace/results/ncu-probe --force-overwrite python /workspace/ncu-probe.py > /workspace/results/ncu-probe.log 2>&1 || true
mkdir -p /workspace/master
tar -xf /workspace/master.tar -C /workspace/master
cd /workspace/master
bash setup_script.sh > /workspace/results/master-build.log 2>&1
cp /workspace/nnue/tests/bench_training.py tests/bench_training.py
# The benchmark is an overlay; leave the production trainer implementation exact.
sed -i '/ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,/d' tests/bench_training.py
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
checkpoint=$(cat /workspace/results/checkpoint-path.txt)
for mode in cached live; do
 opts=()
 test "$mode" = cached && opts=(--cached 8 --batch-cache /workspace/cache-skip0)
 torchrun --standalone --nproc-per-node=4 ddp_launcher.py tests/bench_training.py "${data[@]}" --threads 4 --num-workers 64 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --resume-from-checkpoint "$checkpoint" "${opts[@]}" --output "/workspace/results/master-${mode}.json" > "/workspace/results/master-${mode}.log" 2>&1
done
touch /workspace/results/MASTER_DONE

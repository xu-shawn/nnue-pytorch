#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_ARENA_MAX=1
cd /workspace/nnue
python /workspace/quant_experiment.py --parity > /workspace/results/quant-parity.log 2>&1
mapfile -t data < <(find /workspace/data -name '*.binpack' | sort)
for spec in fused:1 control:0; do
  name=${spec%:*}
  export NNUE_FUSED_FT_QUANT=${spec#*:}
  torchrun --standalone --nproc-per-node=4 ddp_launcher.py /workspace/quant_experiment.py "${data[@]}" --threads 1 --num-workers 32 --warmup 64 --steps 512 --repeats 3 --random-fen-skipping 0 --cached 8 --batch-cache /workspace/cache-skip0 --grouped-l1 --sparse-l1 --ddp-bucket-cap-mb 400 --resume-from-checkpoint /workspace/mature-init.ckpt --output /workspace/results/quant-${name}-cached.json > /workspace/results/quant-${name}-cached.log 2>&1
done
touch /workspace/results/QUANT_DONE

#!/bin/bash
set -euo pipefail
export PATH=/workspace/venv/bin:$PATH OPENBLAS_NUM_THREADS=1
while ! test -f /workspace/results/STEADY_DONE; do sleep 5; done
cd /workspace/nnue
python /workspace/sparse-analysis.py > /workspace/results/sparse-weighted.log 2>&1
touch /workspace/results/SPARSE_DONE

#!/bin/bash
# usage: proxy_run.sh <config letter> <out dir> <data> [extra train.py args...]
# --extra-layer among the extra args sets NNUE_EXTRA_LAYER=1 instead of reaching train.py.
cd "$(dirname "$0")"
read -r name feat stacks dtype eq < <(grep -E "^$1 " proxy_configs.sh)
[ -n "$name" ] || { echo "unknown config $1"; exit 1; }
out=$2; data=$3; shift 3
args=(); layer=0
for a in "$@"; do if [ "$a" = --extra-layer ]; then layer=1; else args+=("$a"); fi; done
mkdir -p "$out"
NNUE_STACKS=$stacks NNUE_PSQ_DTYPE=$dtype NNUE_EXTRA_LAYER=$layer python -u train.py $data --features=$feat --default_root_dir="$out" "${args[@]}" > "$out/train.log" 2>&1
echo "$name ($eq) exit $?"

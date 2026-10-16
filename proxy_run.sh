#!/bin/bash
# usage: proxy_run.sh <config letter> <out dir> <data> [extra train.py args...]
cd "$(dirname "$0")"
read -r name feat stacks dtype eq < <(grep -E "^$1 " proxy_configs.sh)
[ -n "$name" ] || { echo "unknown config $1"; exit 1; }
out=$2; data=$3; shift 3
mkdir -p "$out"
NNUE_STACKS=$stacks NNUE_PSQ_DTYPE=$dtype python -u train.py $data --features=$feat --default_root_dir="$out" "$@" > "$out/train.log" 2>&1
echo "$name ($eq) exit $?"

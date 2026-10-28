"""Layer-stack hit counts on a binpack, through the trainer's data loader (production skip config) and the model's own
routing code (calculate_buckets + stack_key_index). usage: count_stacks.py <binpack> <ls-keys> <features> <out.npy> [raw]
("raw": no position filtering or skipping). Feed the counts to make_ls_remap.py."""
import os, sys
from pathlib import Path
import numpy as np
binpack_arg = os.path.abspath(sys.argv[1])
os.chdir(Path(__file__).resolve().parents[1]); sys.path.insert(0, ".")
import torch
from data_loader.config import DataloaderSkipConfig
from data_loader.stream import create_sparse_batch_stream, fetch_next_sparse_batch, destroy_sparse_batch, destroy_sparse_batch_stream
from model.config import ModelConfig
from model.model import NNUEModel
binpack, keys, feats, out = binpack_arg, *sys.argv[2:5]
out = os.path.abspath(out) if not os.path.isabs(out) else out
m = NNUEModel(feats, ModelConfig(L1=256, ls_keys=keys))
raw_mode = len(sys.argv) > 5 and sys.argv[5] == "raw"
cfg = DataloaderSkipConfig(filtered=False, wld_filtered=False, random_fen_skipping=0, early_fen_skipping=-1,
    soft_early_fen_skipping=0, pc_y0=1.0, pc_y1=1.0, pc_y2=1.0, pc_y3=1.0, pc_y4=1.0) if raw_mode else DataloaderSkipConfig(filtered=True, wld_filtered=True, random_fen_skipping=3, early_fen_skipping=18,
    soft_early_fen_skipping=32, pc_y0=-0.2, pc_y1=0.45, pc_y2=1.0, pc_y3=0.95, pc_y4=0.75,
    ply_x1=0.0, ply_y1=0.025, ply_x2=22.0, ply_y2=0.05, ply_x3=25.5, ply_y3=0.2, ply_x4=29.5, ply_y4=0.8)
s = create_sparse_batch_stream(m.input_feature_name.encode(), 2, [binpack], 16384, False, cfg)
tot = np.zeros(m.num_ls_buckets, np.int64)
with torch.no_grad():
    while True:
        b = fetch_next_sparse_batch(s)
        if not b:
            break
        us, them, wi, bi, o, sc, pc, qb = b.contents.get_tensors("cpu")
        raw = m.calculate_buckets(pc, qb).view(-1) * m.ls_key_radix + m.stack_key_index(us, wi).view(-1)
        tot += np.bincount(raw.numpy(), minlength=m.num_ls_buckets)
        destroy_sparse_batch(b)
destroy_sparse_batch_stream(s)
np.save(out, tot)
n = tot.sum()
print(f"{binpack}: {n} positions, {m.num_ls_buckets} stacks; never seen {int((tot == 0).sum())}")
for f in (1e-7, 1e-6, 1e-5, 1e-4):
    dead = tot < f * n
    print(f"  frequency < {f:g}: {int(dead.sum())} stacks, {100 * tot[dead].sum() / n:.4f}% of positions")

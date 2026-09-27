"""Preserve exact replay inputs and source snapshots before rental cleanup."""
import hashlib
import json
import pathlib
import tarfile
import time

root = pathlib.Path('/workspace')
files = [
    root/'source.tar', root/'master.tar', root/'delivery-update.tar', root/'sparse-update.tar',
    root/'control-fused-ft-kernel.py', root/'final-fused-ft-kernel.py',
    root/'delivered-fused-ft-kernel.py', root/'dense-grouped-linear.py', root/'master/tests/bench_training.py',
    root/'reproduction/lightning_logs/version_0/checkpoints/last.ckpt',
    root/'mature-init.ckpt',
    root/'delivered-resume/lightning_logs/version_0/checkpoints/last.ckpt',
    *sorted((root/'cache-skip0').glob('*')),
]
source_files = [
    'config.py', 'train.py', 'scripts/train_threats_h100.sh',
    'trainer/engine.py', 'trainer/callbacks.py', 'model/nnue.py',
    'model/modules/config.py', 'model/modules/layer_stacks.py',
    'model/modules/stacked_linear.py', 'model/modules/grouped_linear.py',
    'model/modules/feature_transformer/fused_ft_kernel.py',
    'tests/bench_training.py', 'tests/test_grouped_linear.py',
    'tests/test_fused_double_ft.py',
]
def info(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return {'bytes':path.stat().st_size,'sha256':h.hexdigest()}
manifest={'base_revision':'eea071f5ddde5247c84ce7e8598f676ba42de462',
          'master_revision':'eba1ed889d5a6a87d897362c829e7da204fbdfcc',
          'source':{p:info(root/'nnue'/p) for p in source_files},
          'replay_files':{str(p.relative_to(root)):info(p) for p in files}}
(root/'results/source-and-replay-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Manifest written',time.time(),flush=True)
with tarfile.open(root/'replay-inputs.tar.gz','w:gz',compresslevel=1) as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(root)))
archive=info(root/'replay-inputs.tar.gz')
(root/'results/replay-archive.json').write_text(json.dumps(archive,indent=2))
print('Archive complete',archive,time.time(),flush=True)

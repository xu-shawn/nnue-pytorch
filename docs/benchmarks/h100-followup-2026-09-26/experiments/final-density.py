import hashlib
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, '/workspace/nnue')
import torch
from model import NNUE

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('bench', '/workspace/nnue/tests/bench_training.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
model = NNUE(config=bench.threats_config(), max_epoch=4500, num_batches_per_epoch=1024).cuda()
batches = torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt', map_location='cuda', weights_only=True)
saved = {}
model.model.layer_stacks.l1.register_forward_pre_hook(lambda m, args: saved.update(x=args[0].detach()))
results = {}
for name, checkpoint in {
    'initial': '/workspace/mature-init.ckpt',
    'cached_after_1600_steps': '/workspace/mature-warmed.ckpt',
    'live_after_1024_steps': '/workspace/delivered-training/lightning_logs/version_0/checkpoints/last.ckpt',
    'live_resumed_1024_more_steps': '/workspace/delivered-resume/lightning_logs/version_0/checkpoints/last.ckpt',
}.items():
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'])
    row = {'checkpoint': checkpoint, 'global_step': state.get('global_step'), 'epoch': state.get('epoch'), 'batches': []}
    with torch.no_grad():
        for batch in batches:
            us, them, wi, bi, _, _, pc = batch
            model.model(us, them, wi, bi, pc, True, True)
            counts = (saved['x'] != 0).sum(1).float()
            row['batches'].append({'nnz_mean': counts.mean().item(), 'nnz_percentiles': torch.quantile(counts, torch.tensor([0., .25, .5, .75, 1.], device='cuda')).tolist()})
    results[name] = row
pathlib.Path('/workspace/results/final-density.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

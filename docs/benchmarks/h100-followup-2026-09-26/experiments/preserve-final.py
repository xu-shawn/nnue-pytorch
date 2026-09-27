"""Archive the final aggregation delta without rebuilding the earlier replay bundle."""
import hashlib
import json
import pathlib
import tarfile

import torch

root=pathlib.Path('/workspace')
assert (root/'results/AGGREGATE_FINAL_DONE').exists()
checkpoint=root/'aggregate-resume/lightning_logs/version_0/checkpoints/last.ckpt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
summary={k:state.get(k) for k in ('epoch','global_step')}
optimizer=state['state_dict']['optimizer_state_dict']
summary['optimizer_parameter_states']=len(optimizer['state'])
summary['optimizer_steps']=sorted({int(v['step']) for v in optimizer['state'].values()})
summary['scheduler_count']=len(state.get('lr_schedulers', []))
(root/'results/aggregate-checkpoint.json').write_text(json.dumps(summary,indent=2))
del state
def info(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
old=json.loads((root/'results/source-and-replay-manifest.json').read_text())
paths=list(old['source'])+['model/modules/feature_transformer/fused_ft_functions.py',
    'model/modules/feature_transformer/aggregated_ft_kernel.py','tests/test_aggregated_ft.py']
manifest={'base_revision':old['base_revision'],'source':{p:info(root/'nnue'/p) for p in paths}}
files=[root/'aggregate-update.tar',root/'aggregate-format-update.tar',checkpoint]
manifest['delta_files']={str(p.relative_to(root)):info(p) for p in files}
(root/'results/final-source-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'aggregate-replay-delta.tar.gz','w:gz',compresslevel=1) as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(root)))
(root/'results/aggregate-replay-delta.json').write_text(json.dumps(info(root/'aggregate-replay-delta.tar.gz'),indent=2))
with tarfile.open(root/'final-results-aggregate.tgz','w:gz',compresslevel=1) as tar:
    tar.add(root/'results',arcname='results')
print('ARCHIVE COMPLETE',summary,flush=True)

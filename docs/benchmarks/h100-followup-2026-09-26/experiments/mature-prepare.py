import hashlib,importlib.util,json,pathlib,sys
sys.path.insert(0,'/workspace/nnue')
import torch
torch.set_num_threads(1)
from model import NNUE
from model.utils.load_model import load_model
spec=importlib.util.spec_from_file_location('bench','/workspace/nnue/tests/bench_training.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
p=pathlib.Path('/workspace/nn-252f33942263.nnue');sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha.startswith('252f33942263'),sha
cfg=bench.threats_config()
m=NNUE(config=cfg,max_epoch=4500,num_batches_per_epoch=1024)
# Copy weights into existing modules so optimizer parameter references stay valid.
loaded=load_model(str(p),cfg.features,cfg.model_config)
m.model.load_state_dict(loaded.state_dict())
torch.save({'epoch':0,'global_step':0,'state_dict':m.state_dict(),'lr_schedulers':[]},'/workspace/mature-init.ckpt')
metadata={'url':'https://tests.stockfishchess.org/api/nn/nn-252f33942263.nnue','bytes':p.stat().st_size,'sha256':sha,'features':cfg.features,'note':'Serialized mature weights; original AdamW moments and schedule state are unavailable. Fresh optimizer for controlled training benchmarks.'}
pathlib.Path('/workspace/results/mature-net.json').write_text(json.dumps(metadata,indent=2))
print(metadata,flush=True)

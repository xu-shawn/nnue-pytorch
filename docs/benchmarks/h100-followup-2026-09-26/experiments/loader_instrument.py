import atexit, collections, json, os, sys, time
sys.path.insert(0,'/workspace/nnue')
if 'NNUE_GIL_INTERVAL' in os.environ:
    sys.setswitchinterval(float(os.environ['NNUE_GIL_INTERVAL']))
import data_loader.dataset as ds
stats=collections.defaultdict(list)
orig_init=ds.TrainingDataProvider.__init__
def init(self,*args,**kwargs):
    orig_init(self,*args,**kwargs)
    fetch=self.fetch_next
    def measured_fetch(*a):
        start=time.perf_counter();v=fetch(*a);stats['native_fetch_ms'].append((time.perf_counter()-start)*1000);return v
    self.fetch_next=measured_fetch
    destroy=self.destroy_part
    def measured_destroy(*a):
        start=time.perf_counter();v=destroy(*a);stats['native_free_ms'].append((time.perf_counter()-start)*1000);return v
    self.destroy_part=measured_destroy
ds.TrainingDataProvider.__init__=init
from data_loader._native import SparseBatch
orig_tensors=SparseBatch.get_tensors
def get_tensors(self,*args,**kwargs):
    start=time.perf_counter();v=orig_tensors(self,*args,**kwargs);stats['pin_copy_submit_ms'].append((time.perf_counter()-start)*1000);return v
SparseBatch.get_tensors=get_tensors
@atexit.register
def report():
    import statistics
    result={k:{'count':len(v[64:]),'mean':statistics.mean(v[64:]),'p95':sorted(v[64:])[int(len(v[64:])*.95)]} for k,v in stats.items() if len(v)>64}
    with open('/workspace/results/loader-stages-rank'+os.environ.get('RANK','0')+'.json','w') as f:json.dump(result,f,indent=2)
if os.environ.get('NNUE_DETAILED_STAGING') == '1':
    import torch
    import data_loader._native as native
    def detailed_move(t, device, use_pinned_memory=False, dtype=None):
        dtype = t.dtype if dtype is None else dtype
        tag = 'large_' if t.numel() > 1000000 else 'small_'
        start = time.perf_counter()
        out = torch.empty(t.shape, dtype=dtype, layout=t.layout, device='cpu', pin_memory=use_pinned_memory)
        stats[tag+'allocate_ms'].append((time.perf_counter()-start)*1000)
        start = time.perf_counter()
        out.copy_(t)
        stats[tag+'copy_ms'].append((time.perf_counter()-start)*1000)
        start = time.perf_counter()
        result = out.to(device=device, non_blocking=use_pinned_memory)
        stats[tag+'submit_ms'].append((time.perf_counter()-start)*1000)
        return result
    native._pin_and_move = detailed_move
import runpy
runpy.run_path('/workspace/nnue/tests/bench_training.py',run_name='__main__')

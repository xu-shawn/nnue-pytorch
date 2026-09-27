"""Sweep FT backward on a trained checkpoint and an identical real batch."""
import itertools, json, pathlib, random, statistics, sys
sys.path.insert(0, '/workspace/nnue')
import numpy as np
import torch
from model import NNUE
from model.modules.feature_transformer import fused_ft_kernel as fk
from model.modules.feature_transformer.fused_ft_functions import FusedDoubleFtFunction
import importlib.util
spec = importlib.util.spec_from_file_location('bench', '/workspace/nnue/tests/bench_training.py')
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
torch.set_num_threads(1); torch.manual_seed(42)
model = NNUE(config=bench.threats_config(), max_epoch=4500, num_batches_per_epoch=1024).cuda()
checkpoint = pathlib.Path('/workspace/results/checkpoint-path.txt').read_text().strip()
model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['state_dict'])
batch = torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt', map_location='cuda', weights_only=True)[0]
us,them,wi,bi,*_ = batch
ft = model.model.input
with torch.no_grad(): weight,bias = ft.merged_weight_and_bias(False)
l1=1024; ninputs=weight.shape[0]
maxact=ft.quantization.max_ft_activation
class Ctx:
    def save_for_backward(self,*args): self.saved_tensors=args
ctx=Ctx()
out=FusedDoubleFtFunction.forward(ctx,us,them,wi,bi,weight,bias,maxact,l1)
clamped=ctx.saved_tensors[-1]
gl=torch.randn_like(out)*1e-4
source=pathlib.Path(fk.__file__).read_text()
base_args=(us.data_ptr(),them.data_ptr(),wi.data_ptr(),bi.data_ptr(),weight.data_ptr(),bias.data_ptr(),np.float32(maxact),gl.data_ptr(),clamped.data_ptr())

def build(threads,tile,shards,hotstart,hotend,colsplit):
    ns=fk.__dict__.copy()
    variant=source.replace('min(l1_half, 1024)',str(threads))
    if colsplit:
        pos=variant.index('void fused_double_ft_backward(')
        head,body=variant[:pos],variant[pos:]
        body=body.replace('const uint32_t tid = threadIdx.x;', 'const uint32_t tid = threadIdx.x + blockIdx.y * blockDim.x;')
        body=body.replace('str(num_threads)', 'str(l1_half)')
        variant=head+body
    if shards>1:
        variant=variant.replace('float* __restrict__ grad_bias,', 'float* __restrict__ grad_bias,\n          float* __restrict__ hot_grad_weight,')
        for pov in ['w','b']:
            target=f'atomicAdd(&grad_weight[{pov}_idx * output_size + col],'
            replacement=f'''float* target = ({pov}_idx >= {hotstart} && {pov}_idx < {hotend}) ? hot_grad_weight + ((blockIdx.x % {shards}) * {hotend-hotstart} + {pov}_idx - {hotstart}) * output_size : grad_weight + {pov}_idx * output_size;
                atomicAdd(&target[col],'''
            variant=variant.replace(target,replacement)
            variant=variant.replace(f'atomicAdd(&grad_weight[{pov}_idx * output_size + col + l1_half],', 'atomicAdd(&target[col + l1_half],')
    exec(variant,ns)
    kernel=ns['make_fused_double_ft_backward_kernel'](wi.shape[1],l1,tile_size=tile)
    gw=torch.empty_like(weight); gb=torch.empty_like(bias)
    hot=torch.empty(shards,hotend-hotstart,l1,device='cuda') if shards>1 else None
    args=base_args+(gw.data_ptr(),gb.data_ptr())+((hot.data_ptr(),) if shards>1 else ())+(np.int32(len(us)),np.int32(l1))
    grid=((len(us)+tile-1)//tile,512//threads if colsplit else 1)
    def run():
        gw.zero_(); gb.zero_()
        if hot is not None: hot.zero_()
        kernel(grid=grid,args=args)
        if hot is not None: gw[hotstart:hotend].copy_(hot.sum(0))
        return gw,gb
    return run

def timing(run):
    for _ in range(3):run()
    times=[]
    for _ in range(3):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):run()
        end.record(); end.synchronize(); times.append(start.elapsed_time(end)/10)
    return statistics.median(times)
reference=build(512,4,1,0,0,False)
refgw,refgb=[x.clone() for x in reference()]
configs=[]
for threads,tile,split in itertools.product([128,256,512],[1,4,8,16],[False,True]):
    if split and threads==512:continue
    configs.append((threads,tile,1,0,0,split))
for shards in [2,4,8,16]:
    for hotstart,hotend in [(59808,64368),(0,ninputs)]:
        configs.append((512,4,shards,hotstart,hotend,False))
random.Random(42).shuffle(configs)
results=[]
for config in configs:
    run=build(*config)
    gw,gb=run()
    torch.testing.assert_close(gw,refgw,atol=3e-7,rtol=3e-4)
    torch.testing.assert_close(gb,refgb,atol=3e-7,rtol=3e-4)
    row=dict(zip(['threads','tile','shards','hotstart','hotend','colsplit'],config));row['ms']=timing(run)
    results.append(row);print(json.dumps(row),flush=True)
    del run
print('BASELINE',timing(reference),flush=True)
pathlib.Path('/workspace/results/tune-backward.json').write_text(json.dumps(results,indent=2))
from vector_kernel import make_vector_backward
for tile in [1,2,4,8,16]:
    kernel=make_vector_backward(wi.shape[1],l1,tile)
    gw=torch.empty_like(weight);gb=torch.empty_like(bias)
    args=base_args+(gw.data_ptr(),gb.data_ptr(),np.int32(len(us)),np.int32(l1))
    def run():
        gw.zero_();gb.zero_()
        kernel(grid=((len(us)+tile-1)//tile,),args=args)
        return gw,gb
    gw,gb=run()
    torch.testing.assert_close(gw,refgw,atol=3e-7,rtol=3e-4)
    torch.testing.assert_close(gb,refgb,atol=3e-7,rtol=3e-4)
    row={'vector':4,'tile':tile,'ms':timing(run)};results.append(row);print(json.dumps(row),flush=True)
pathlib.Path('/workspace/results/tune-backward.json').write_text(json.dumps(results,indent=2))

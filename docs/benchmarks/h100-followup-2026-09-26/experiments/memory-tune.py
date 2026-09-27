"""FT traffic/atomic hypotheses on the mature network and actual loss gradients."""
import importlib.util,json,pathlib,statistics,sys
sys.path.insert(0,'/workspace/nnue')
import cupy as cp
import numpy as np
import torch
from model import NNUE
from model.modules.feature_transformer import fused_ft_kernel as fk
from model.modules.feature_transformer import composed_feature_transformer as composed
from model.modules.feature_transformer.fused_ft_functions import FusedDoubleFtFunction

torch.set_num_threads(1);torch.manual_seed(42)
spec=importlib.util.spec_from_file_location('bench','/workspace/nnue/tests/bench_training.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
model=NNUE(config=bench.threats_config(True,True),max_epoch=4500,num_batches_per_epoch=1024).cuda()
model.load_state_dict(torch.load('/workspace/mature-init.ckpt',map_location='cuda',weights_only=False)['state_dict'])
batch=torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt',map_location='cuda',weights_only=True)[0]
us,them,wi,bi,*_=batch
saved={};original=composed.double_feature_transform
def capture(*args,**kwargs):
    out=original(*args,**kwargs);out.retain_grad();saved['raw']=out;return out
composed.double_feature_transform=capture
loss=model.compute_loss(batch,0);loss.backward()
gl=saved['raw'].grad.detach().contiguous()
with torch.no_grad():weight,bias=model.model.input.merged_weight_and_bias(True)
maxact=model.model.input.quantization.max_ft_activation
class Ctx:
    def save_for_backward(self,*args):self.saved_tensors=args
ctx=Ctx();out=FusedDoubleFtFunction.forward(ctx,us,them,wi,bi,weight,bias,maxact,1024)
clamped=ctx.saved_tensors[-1]
payload={k:v.cpu() for k,v in dict(us=us,them=them,wi=wi,bi=bi,weight=weight,bias=bias,grad=gl).items()}
payload['maxact']=maxact
torch.save(payload,'/workspace/counter-inputs.pt')
print('INPUTS',weight.shape,wi.shape,'loss',loss.item(),'nonzero upstream',(gl!=0).float().mean().item(),flush=True)

def raw_kernel(fn):
    return next(cell.cell_contents for cell in fn.__closure__ if isinstance(cell.cell_contents,cp.RawKernel))
baseargs=(us.data_ptr(),them.data_ptr(),wi.data_ptr(),bi.data_ptr(),weight.data_ptr(),bias.data_ptr(),np.float32(maxact),gl.data_ptr(),clamped.data_ptr())
def build(tile,register=False,partial=False,outer=False):
    code=raw_kernel(fk.make_fused_double_ft_backward_kernel(wi.shape[1],1024,tile)).code
    if register or partial:
        start=code.index('    __shared__ float shared_grad_bias')
        end=code.index('    for (int t =',start)
        code=code[:start]+'    float bias0=0.0f,bias1=0.0f;\n'+code[end:]
        code=code.replace('shared_grad_bias[col]           +=','bias0 +=').replace('shared_grad_bias[col + l1_half] +=','bias1 +=')
        start=code.index('    __syncthreads();')
        code=code[:start]+('''
    grad_bias[tile_idx*1024+tid]=bias0;
    grad_bias[tile_idx*1024+tid+512]=bias1;
}
''' if partial else '''
    if(bias0!=0.0f)atomicAdd(&grad_bias[tid],bias0);
    if(bias1!=0.0f)atomicAdd(&grad_bias[tid+512],bias1);
}
''')
    if outer:
        for pov in ['w','b']:
            start=code.index('            for(int k=0;',code.index(f'float g_{pov}0'))
            if pov=='b':start=code.index('            for(int k=0;',code.index('int b_idx =')-80)
            end=code.index('\n            }',start)+len('\n            }')
            old=code[start:end]
            code=code[:start]+f'            if(g_{pov}0!=0.0f || g_{pov}1!=0.0f){{\n'+old+'\n            }'+code[end:]
    kernel=cp.RawKernel(code,'fused_double_ft_backward');kernel.compile()
    gw=torch.empty_like(weight);gb=torch.empty_like(bias)
    tiles=(len(us)+tile-1)//tile
    temp=torch.empty((tiles,1024),device='cuda') if partial else gb
    args=baseargs+(gw.data_ptr(),temp.data_ptr(),np.int32(len(us)),np.int32(1024))
    def run():
        gw.zero_()
        if not partial:gb.zero_()
        kernel((tiles,4),(128,),args)
        if partial:torch.sum(temp,dim=0,out=gb)
        return gw,gb
    return run,code

def timing(fn,n=10):
    for _ in range(5):fn()
    vals=[]
    for _ in range(3):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(n):fn()
        b.record();b.synchronize();vals.append(a.elapsed_time(b)/n)
    return statistics.median(vals)

reference,_=build(4)
expected=[t.clone() for t in reference()]
results={'backward':[],'duplicates':[]}
for tile in [2,4,8,16]:
    for register,partial,outer in [(False,False,False),(True,False,False),(True,True,False),(True,False,True)]:
        fn,code=build(tile,register,partial,outer)
        for actual,ref in zip(fn(),expected):torch.testing.assert_close(actual,ref,atol=3e-7,rtol=3e-4)
        row=dict(tile=tile,register_bias=register,partial_bias=partial,outer_zero_guard=outer,ms=timing(fn))
        results['backward'].append(row);print(row,flush=True)
        pathlib.Path('/workspace/results/memory-tune.json').write_text(json.dumps(results,indent=2))

# A cheap duplicate count establishes whether aggregation can reduce atomics.
white=wi.cpu().numpy();black=bi.cpu().numpy()
for tile in [2,4,8,16]:
    ids=np.stack([white,black],axis=1).reshape(len(us)//tile,-1)
    ids=np.sort(ids,axis=1)
    valid=ids>=0
    unique=valid[:,0].sum()+((ids[:,1:]!=ids[:,:-1])&valid[:,1:]).sum()
    total=valid.sum()
    row={'tile':tile,'indices':int(total),'unique':int(unique),'potential_atomic_reduction':float(1-unique/total)}
    results['duplicates'].append(row);print(row,flush=True)

fw=raw_kernel(fk.make_fused_double_ft_forward_kernel(wi.shape[1],1024)) if False else None
# Forward wrappers use _kernel_with_threads, whose closure contains the RawKernel.
fw=raw_kernel(fk.make_fused_double_ft_forward_kernel(wi.shape[1],1024))
quant=lambda z: model.model.input.quantization.fake_quantize_ft_act(z)*model.model.input.quantization.l0_correction_factor
post=torch.compile(quant)
post(out)
code=fw.code
code=code.replace('= l0_w0 * l0_w1;', '= floorf(__fadd_rn(__fmul_rn(__fmul_rn(l0_w0,l0_w1),128.0f),1.0e-5f))*(1.0f/128.0f);')
code=code.replace('= l0_b0 * l0_b1;', '= floorf(__fadd_rn(__fmul_rn(__fmul_rn(l0_b0,l0_b1),128.0f),1.0e-5f))*(1.0f/128.0f);')
fused=cp.RawKernel(code,'fused_double_ft_forward');fused.compile()
y=torch.empty_like(out);cl=torch.empty_like(clamped)
fa=(us.data_ptr(),them.data_ptr(),wi.data_ptr(),bi.data_ptr(),weight.data_ptr(),bias.data_ptr(),np.float32(maxact),y.data_ptr(),cl.data_ptr(),np.int32(1024))
def baseline_forward():
    fw((len(us),),(256,),fa);return post(y)
def fused_forward():
    fused((len(us),),(256,),fa);return y
ref=baseline_forward().clone();actual=fused_forward()
torch.testing.assert_close(actual,ref,atol=0,rtol=0)
torch.testing.assert_close(cl,clamped,atol=0,rtol=0)
results['forward']={'separate_ms':timing(baseline_forward),'fused_quant_ms':timing(fused_forward)}
print(results['forward'],flush=True)
pathlib.Path('/workspace/results/memory-tune.json').write_text(json.dumps(results,indent=2))

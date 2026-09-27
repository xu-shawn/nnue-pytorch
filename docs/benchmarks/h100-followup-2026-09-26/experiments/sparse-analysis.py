import importlib.util,json,pathlib,statistics,sys
sys.path.insert(0,'/workspace/nnue')
sys.path.insert(0,'/workspace')
import torch
from model import NNUE
from model.modules import grouped_linear as grouped
import sparse_weighted as sparse

torch.set_num_threads(1)
torch.manual_seed(42)
spec=importlib.util.spec_from_file_location('bench','/workspace/nnue/tests/bench_training.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
model=NNUE(config=bench.threats_config(),max_epoch=4500,num_batches_per_epoch=1024).cuda()
model.load_state_dict(torch.load('/workspace/mature-init.ckpt',map_location='cuda',weights_only=False)['state_dict'])
batch=torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt',map_location='cuda',weights_only=True)[0]
saved={}
layer=model.model.layer_stacks.l1
layer.register_forward_pre_hook(lambda m,args:saved.update(args=args))

def timing(fn,iters=30):
    for _ in range(5):fn()
    values=[]
    for _ in range(3):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):fn()
        b.record();b.synchronize();values.append(a.elapsed_time(b)/iters)
    return statistics.median(values)

with torch.no_grad():
    us,them,wi,bi,_,_,pc=batch
    model.model(us,them,wi,bi,pc,True,True)
    x,idx,*_=saved['args'];x=x.contiguous();idx=idx.flatten().to(torch.int64).contiguous()
    q=model.model.quantization
    w=q.fake_quantize_weights(layer.linear.weight+layer.factorized_linear.weight.repeat(8,1),'ls_l1_weight').contiguous()
    b=q.fake_quantize_weights(layer.linear.bias+layer.factorized_linear.bias.repeat(8),'ls_l1_bias').contiguous()
    reference=lambda:torch.nn.functional.linear(x,w,b).reshape(-1,8,32)[torch.arange(len(x),device=x.device),idx]
    ref=torch.compile(reference)
    expected=ref()
    rows,counts=grouped._route_rows(idx)
    wt=sparse.transpose(w)
    results={'bullet_revision':'6b1795767d6a4080cea6eb99fbba9f1f462a8924','nnz_mean':(x!=0).sum(1).float().mean().item(),'reference_ms':timing(ref),'grouped_forward_ms':timing(lambda:grouped._forward(x,w,b,rows,counts,32,64,64)),'transpose_ms':timing(lambda:sparse.transpose(w)),'variants':[]}
    for threads in [32,64,128,256]:
        for mode in ['ballot','compact']:
            actual=sparse.forward(x,wt,b,idx,threads,mode)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            row={'threads':threads,'mode':mode,'kernel_ms':timing(lambda:sparse.forward(x,wt,b,idx,threads,mode)),'with_transpose_ms':timing(lambda:sparse.forward(x,sparse.transpose(w),b,idx,threads,mode))}
            results['variants'].append(row);print(row,flush=True)
        packed=sparse.pack(x,threads)
        for vector in [1,2,4]:
            actual=sparse.forward(x,wt,b,idx,threads,'packed',vector,packed)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            row={'threads':threads,'mode':'packed','vector':vector,'pack_ms':timing(lambda:sparse.pack(x,threads)),'kernel_ms':timing(lambda:sparse.forward(x,wt,b,idx,threads,'packed',vector,packed)),'with_pack_transpose_ms':timing(lambda:sparse.forward(x,sparse.transpose(w),b,idx,threads,'packed',vector))}
            results['variants'].append(row);print(row,flush=True)
        pathlib.Path('/workspace/results/sparse-weighted.json').write_text(json.dumps(results,indent=2))
    print(results,flush=True)

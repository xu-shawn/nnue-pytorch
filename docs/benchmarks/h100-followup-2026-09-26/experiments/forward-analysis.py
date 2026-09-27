import importlib.util,json,os,pathlib,statistics,sys
sys.path.insert(0,'/workspace/nnue')
sys.path.insert(0,'/workspace')
import torch
from model import NNUE
from selected_linear import forward,dx
spec=importlib.util.spec_from_file_location('bench','/workspace/nnue/tests/bench_training.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
torch.set_num_threads(1);torch.manual_seed(42)
model=NNUE(config=bench.threats_config(),max_epoch=4500,num_batches_per_epoch=1024).cuda()
checkpoint=os.environ.get('NNUE_CHECKPOINT') or pathlib.Path('/workspace/results/checkpoint-path.txt').read_text().strip()
model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict'])
batches=torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt',map_location='cuda',weights_only=True)
ls=model.model.layer_stacks
saved={}
for name,layer in [('l1',ls.l1),('l2',ls.l2),('output',ls.output)]:
 layer.register_forward_pre_hook(lambda m,args,name=name:saved.update({name:tuple(t.detach() if isinstance(t,torch.Tensor) else t for t in args)}))
results={'checkpoint':checkpoint,'layers':{},'stages':{}}

def timing(fn,iters=30):
 for _ in range(5):fn()
 values=[]
 for _ in range(3):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(iters):fn()
  b.record();b.synchronize();values.append(a.elapsed_time(b)/iters)
 return statistics.median(values)

with torch.no_grad():
 for batch in batches[:1]:
  us,them,wi,bi,_,_,pc=batch
  model.model(us,them,wi,bi,pc,True,True)
 for name,layer in [('l1',ls.l1),('l2',ls.l2),('output',ls.output)]:
  x,idx,*_=saved[name];idx=idx.to(torch.int64).flatten().contiguous();x=x.contiguous()
  w=layer.linear.weight;b=layer.linear.bias
  if name=='l1':
   w=w+layer.factorized_linear.weight.repeat(8,1);b=b+layer.factorized_linear.bias.repeat(8)
  q=model.model.quantization
  w=q.fake_quantize_weights(w,f'{layer.layer_key}_weight').contiguous();b=q.fake_quantize_weights(b,f'{layer.layer_key}_bias').contiguous()
  k=x.shape[1];n=layer.out_features
  def reference(x,w,b,idx):return torch.nn.functional.linear(x,w,b).reshape(-1,8,n)[torch.arange(len(x),device=x.device),idx]
  ref=torch.compile(reference)
  y=ref(x,w,b,idx)
  counts=(x!=0).sum(1).float();grad=torch.randn_like(y)*1e-4
  fullgy=torch.zeros((len(x),8,n),device=x.device);fullgy[torch.arange(len(x),device=x.device),idx]=grad
  fulldy=fullgy.reshape(len(x),-1)
  row={'shape':list(x.shape),'outputs':n,'nnz_mean':counts.mean().item(),'nnz_percentiles':torch.quantile(counts,torch.tensor([0.,.25,.5,.75,1.],device='cuda')).tolist(),'bucket_counts':torch.bincount(idx,minlength=8).tolist(),'reference_forward_ms':timing(lambda:ref(x,w,b,idx)),'reference_dx_ms':timing(lambda:fulldy@w),'reference_dw_ms':timing(lambda:fulldy.T@x),'variants':[]}
  print("INPUT",name,json.dumps(row),flush=True)
  reference_dx=fulldy@w
  for threads in [64,128,256]:
   for sparse in [False,True]:
    if sparse and k%threads:continue
    out=forward(x,w,b,idx,n,threads,sparse)
    torch.testing.assert_close(out,y,atol=2e-6,rtol=2e-5)
    ms=timing(lambda:forward(x,w,b,idx,n,threads,sparse))
    row['variants'].append({'threads':threads,'compact_nnz':sparse,'forward_ms':ms})
   dout=dx(grad,w,idx,k,threads)
   torch.testing.assert_close(dout,reference_dx,atol=2e-7,rtol=2e-5)
   row['variants'].append({'threads':threads,'dx_ms':timing(lambda:dx(grad,w,idx,k,threads))})
  print("DIRECT",name,json.dumps(row["variants"]),flush=True)
  import grouped_linear as grouped
  rows,bc=grouped.route(idx)
  row['routing_ms']=timing(lambda:grouped.route(idx))
  if name=='l1':
   for bm in [16,32,64]:
    for bk in [32,64]:
     out=grouped.forward(x,w,b,rows,bc,n,bm,bk)
     torch.testing.assert_close(out,y,atol=2e-6,rtol=2e-5)
     row['variants'].append({'grouped_forward':[bm,bk],'ms':timing(lambda:grouped.forward(x,w,b,rows,bc,n,bm,bk))})
   for bm,bk in [(16,64),(32,64),(32,128),(64,64)]:
    out=grouped.dx(grad,w,rows,bc,k,bm,bk)
    torch.testing.assert_close(out,reference_dx,atol=2e-7,rtol=2e-5)
    row['variants'].append({'grouped_dx':[bm,bk],'ms':timing(lambda:grouped.dx(grad,w,rows,bc,k,bm,bk))})
   refdw=fulldy.T@x;refdb=fulldy.sum(0)
   for split,bm,bk in [(4,32,32),(8,32,32),(16,32,32),(8,32,64),(8,64,32),(16,16,32)]:
    ow,ob=grouped.dw(x,grad,rows,bc,split,bm,bk)
    torch.testing.assert_close(ow,refdw,atol=2e-7,rtol=2e-4)
    torch.testing.assert_close(ob,refdb,atol=2e-7,rtol=2e-4)
    row['variants'].append({'grouped_dw':[split,bm,bk],'ms':timing(lambda:grouped.dw(x,grad,rows,bc,split,bm,bk))})
  results['layers'][name]=row
  pathlib.Path(os.environ.get('NNUE_FORWARD_OUTPUT','/workspace/results/forward-analysis.json')).write_text(json.dumps(results,indent=2))
  print(name,json.dumps(row),flush=True)
 # Forward-stage timings use compiled subgraphs, excluding their input creation.
 ft=model.model.input
 merged_fn=torch.compile(ft.merged_weight_and_bias)
 weight,bias=merged_fn(True)
 from model.modules.feature_transformer.double_ft_functions import double_feature_transform
 raw=lambda:double_feature_transform(us,them,wi,bi,weight,bias,ft.quantization.max_ft_activation,1024,'fused')
 l0=raw()
 post=torch.compile(lambda z:ft.quantization.fake_quantize_ft_act(z)*ft.quantization.l0_correction_factor)
 results['stages']['merge_and_quantize_ft_weights_ms']=timing(lambda:merged_fn(True))
 results['stages']['ft_forward_ms']=timing(raw)
 results['stages']['ft_output_quantize_ms']=timing(lambda:post(l0))
 from model.modules.feature_transformer import fused_ft_kernel as fk
 from vector_forward import make_forward
 import numpy as np
 output=torch.empty_like(l0);clamps=torch.empty((len(us),2048),device='cuda')
 args=(us.data_ptr(),them.data_ptr(),wi.data_ptr(),bi.data_ptr(),weight.data_ptr(),bias.data_ptr(),np.float32(ft.quantization.max_ft_activation),output.data_ptr(),clamps.data_ptr(),np.int32(1024))
 source=pathlib.Path(fk.__file__).read_text()
 variants=[]
 for threads in [32,64,128,256,512]:
  ns=fk.__dict__.copy();exec(source.replace('_FORWARD_THREADS = 128',f'_FORWARD_THREADS = {threads}'),ns)
  kernel=ns['make_fused_double_ft_forward_kernel'](wi.shape[1],1024)
  kernel(grid=(len(us),),args=args)
  torch.testing.assert_close(output,l0,atol=2e-6,rtol=2e-5)
  variants.append({'scalar_threads':threads,'ms':timing(lambda:kernel(grid=(len(us),),args=args),10)})
 reference_clamps=clamps.clone()
 for threads in [32,64,128]:
  kernel=make_forward(wi.shape[1],threads);kernel(grid=(len(us),),args=args)
  torch.testing.assert_close(output,l0,atol=2e-6,rtol=2e-5)
  torch.testing.assert_close(clamps,reference_clamps,atol=2e-6,rtol=2e-5)
  variants.append({'vector_threads':threads,'ms':timing(lambda:kernel(grid=(len(us),),args=args),10)})
 results['ft_forward_variants']=variants
 print('STAGES',results['stages'],variants,flush=True)
pathlib.Path(os.environ.get('NNUE_FORWARD_OUTPUT','/workspace/results/forward-analysis.json')).write_text(json.dumps(results,indent=2))

"""Isolated mature-network forward/backward experiments."""
import os,pathlib,re,runpy,sys
sys.path.insert(0,'/workspace/nnue');sys.path.insert(0,'/workspace')
import torch
from model.modules.feature_transformer import fused_ft_kernel as fk,fused_ft_functions as ff
if 'NNUE_FWD_THREADS' in os.environ or os.environ.get('NNUE_SKIP_ZERO')=='1':
 source=pathlib.Path('/workspace/final-fused-ft-kernel.py').read_text()
 if 'NNUE_FWD_THREADS' in os.environ:source=source.replace('_FORWARD_THREADS = 128','_FORWARD_THREADS = '+os.environ['NNUE_FWD_THREADS'])
 if os.environ.get('NNUE_SKIP_ZERO')=='1':source=re.sub(r'atomicAdd\((&grad_weight\[.*?\]),\s*(g_[wb][01])\);',r'if (\2 != 0.0f) atomicAdd(\1, \2);',source)
 ns=fk.__dict__.copy();exec(source,ns)
 ff.make_fused_double_ft_forward_kernel=ns['make_fused_double_ft_forward_kernel']
 ff.make_fused_double_ft_backward_kernel=ns['make_fused_double_ft_backward_kernel']
if os.environ.get('NNUE_GROUPED')=='1':
 import grouped_linear as gl
 from model.modules.stacked_linear import FactorizedStackedLinear
 class Selected(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x,w,b,indices):
   x=x.contiguous();w=w.contiguous();indices=indices.flatten().to(torch.int64).contiguous()
   rows,counts=gl.route(indices)
   ctx.save_for_backward(x,w,rows,counts)
   return gl.forward(x,w,b.contiguous(),rows,counts,32,64,64)
  @staticmethod
  def backward(ctx,grad):
   x,w,rows,counts=ctx.saved_tensors;grad=grad.contiguous()
   dx=gl.dx(grad,w,rows,counts,1024,64,64)
   dw,db=gl.dw(x,grad,rows,counts,16,32,32)
   return dx,dw,db,None
 @torch.compiler.disable
 def selected(x,w,b,indices):return Selected.apply(x,w,b,indices)
 original=FactorizedStackedLinear.forward
 def fwd(self,x,indices,fake_quantize_weights=False):
  if x.shape[1]!=1024 or self.out_features!=32 or self.count!=8:return original(self,x,indices,fake_quantize_weights)
  w=self.linear.weight+self.factorized_linear.weight.repeat(self.count,1)
  b=self.linear.bias+self.factorized_linear.bias.repeat(self.count)
  if fake_quantize_weights:
   w=self.quantization.fake_quantize_weights(w,f'{self.layer_key}_weight')
   b=self.quantization.fake_quantize_weights(b,f'{self.layer_key}_bias')
  return selected(x,w,b,indices)
 FactorizedStackedLinear.forward=fwd
if os.environ.get('NNUE_GROUPED')=='1' and os.environ.get('NNUE_SKIP_ZERO')=='1':
 sys.argv.extend(['--save-checkpoint','/workspace/mature-warmed.ckpt'])
runpy.run_path('/workspace/nnue/tests/bench_training.py',run_name='__main__')

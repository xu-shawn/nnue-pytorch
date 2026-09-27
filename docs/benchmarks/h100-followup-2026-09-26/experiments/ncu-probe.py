import sys
sys.path.insert(0, '/workspace/nnue')
import torch
from model.modules.feature_transformer.double_ft_functions import double_feature_transform
batch, width, active, inputs = 32768, 1024, 96, 1000
us=torch.ones(batch,1,device='cuda'); them=torch.zeros_like(us)
wi=torch.randint(0,inputs,(batch,active),device='cuda',dtype=torch.int32)
bi=torch.randint(0,inputs,(batch,active),device='cuda',dtype=torch.int32)
w=torch.randn(inputs,width,device='cuda',requires_grad=True)*.001; w.retain_grad()
b=torch.zeros(width,device='cuda',requires_grad=True)
x=double_feature_transform(us,them,wi,bi,w,b,255/256,width,'fused')
x.sum().backward(); torch.cuda.synchronize()

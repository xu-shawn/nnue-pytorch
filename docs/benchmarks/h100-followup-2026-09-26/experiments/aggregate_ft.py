"""Measure tile-local feature aggregation including preprocessing and gradient zeroing."""
import json
import pathlib
import statistics
import sys
sys.path.insert(0, '/workspace/nnue')
import cupy as cp
import numpy as np
import torch
from model.modules.feature_transformer import fused_ft_kernel as fk
from model.modules.feature_transformer.fused_ft_functions import FusedDoubleFtFunction

torch.set_num_threads(1)
p = torch.load('/workspace/counter-inputs.pt', map_location='cuda', weights_only=True)
us, them, wi, bi, weight, bias, gl = [p[k] for k in ['us','them','wi','bi','weight','bias','grad']]
B, A = wi.shape
for indices in (wi, bi):
    ordered = indices.sort(dim=1).values
    assert not ((ordered[:,1:] == ordered[:,:-1]) & (ordered[:,1:] >= 0)).any(), 'bitmask prototype requires no duplicate feature within a row'
class Ctx:
    def save_for_backward(self, *args): self.saved_tensors = args
ctx = Ctx()
out = FusedDoubleFtFunction.forward(ctx, us, them, wi, bi, weight, bias, p['maxact'], 1024)
cl = ctx.saved_tensors[-1]
gw, gb = torch.empty_like(weight), torch.empty_like(bias)
raw = lambda fn: next(c.cell_contents for c in fn.__closure__ if isinstance(c.cell_contents, cp.RawKernel))
back = raw(fk.make_fused_double_ft_backward_kernel(A, 1024))
forward = raw(fk.make_fused_double_ft_forward_kernel(A, 1024))
pathlib.Path('/workspace/results/ft-forward.cu').write_text(forward.code)
pathlib.Path('/workspace/results/ft-backward.cu').write_text(back.code)
args = (us.data_ptr(), them.data_ptr(), wi.data_ptr(), bi.data_ptr(), weight.data_ptr(), bias.data_ptr(), np.float32(p['maxact']), gl.data_ptr(), cl.data_ptr(), gw.data_ptr(), gb.data_ptr(), np.int32(B), np.int32(1024))
def baseline():
    gw.zero_(); gb.zero_(); back((B//4,4), (128,), args)
baseline()
refw, refb = gw.clone(), gb.clone()
def timing(fn):
    for _ in range(5): fn()
    vals=[]
    for _ in range(3):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
        for _ in range(10):fn()
        b.record(); b.synchronize(); vals.append(a.elapsed_time(b)/10)
    return statistics.median(vals)
results={'baseline_ms':timing(baseline),'variants':[]}
print(results,flush=True)
for tile in (4,8,16):
    maxids=2*tile*A
    h=1 << (maxids-1).bit_length()
    tiles=B//tile
    ids=torch.empty((tiles,maxids),device='cuda',dtype=torch.int32)
    masks=torch.empty_like(ids)
    counts=torch.empty(tiles,device='cuda',dtype=torch.int32)
    code=f'#define T {tile}\n#define A {A}\n#define H {h}\n#define M {maxids}\n'+r'''
extern "C" __global__ void pack(const int* w,const int* b,int* ids,unsigned* masks,int* counts){
    extern __shared__ unsigned mem[];
    int* keys=(int*)mem;
    unsigned* bits=mem+H;
    __shared__ int n;
    if(threadIdx.x==0)n=0;
    for(int i=threadIdx.x;i<H;i+=blockDim.x){keys[i]=-1;bits[i]=0;}
    __syncthreads();
    for(int i=threadIdx.x;i<M;i+=blockDim.x){
        int row=i/A, k=i%A;
        int pos=blockIdx.x*T+row/2;
        int id=(row&1)?b[pos*A+k]:w[pos*A+k];
        if(id<0)continue;
        unsigned slot=((unsigned)id*2654435761u)&(H-1);
        while(true){
            int old=atomicCAS(keys+slot,-1,id);
            if(old==-1||old==id){atomicOr(bits+slot,1u<<row);break;}
            slot=(slot+1)&(H-1);
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<H;i+=blockDim.x){
        if(keys[i]>=0){int offset=atomicAdd(&n,1);ids[blockIdx.x*M+offset]=keys[i];masks[blockIdx.x*M+offset]=bits[i];}
    }
    __syncthreads();
    if(threadIdx.x==0)counts[blockIdx.x]=n;
}
extern "C" __global__ void aggregate(const float* us,const float* them,const float* gl,const float* cl,
    float maxact,const int* ids,const unsigned* masks,const int* counts,float* gw,float* gb){
    __shared__ float g[2*T][256];
    int tid=threadIdx.x,col=tid+128*blockIdx.y;
    float bias0=0,bias1=0;
    for(int t=0;t<T;++t){
        int row=blockIdx.x*T+t;
        float w0=cl[row*2048+col],w1=cl[row*2048+512+col];
        float b0=cl[row*2048+1024+col],b1=cl[row*2048+1536+col];
        float d0=gl[row*1024+col],d1=gl[row*1024+512+col];
        float dw0=(w0==0||w0==maxact)?0:d0*w1;
        float dw1=(w1==0||w1==maxact)?0:d0*w0;
        float db0=(b0==0||b0==maxact)?0:d1*b1;
        float db1=(b1==0||b1==maxact)?0:d1*b0;
        float u=us[row],v=them[row];
        float gw0=u*dw0+v*db0, gw1=u*dw1+v*db1;
        float gb0=v*dw0+u*db0, gb1=v*dw1+u*db1;
        g[2*t][tid]=gw0;g[2*t][tid+128]=gw1;
        g[2*t+1][tid]=gb0;g[2*t+1][tid+128]=gb1;
        bias0+=gw0+gb0;bias1+=gw1+gb1;
    }
    __syncthreads();
    int n=counts[blockIdx.x];
    for(int i=0;i<n;++i){
        int id=ids[blockIdx.x*M+i];
        unsigned mask=masks[blockIdx.x*M+i];
        float v0=0,v1=0;
        while(mask){int r=__ffs(mask)-1;mask&=mask-1;v0+=g[r][tid];v1+=g[r][tid+128];}
        if(v0!=0)atomicAdd(gw+id*1024+col,v0);
        if(v1!=0)atomicAdd(gw+id*1024+512+col,v1);
    }
    if(bias0!=0)atomicAdd(gb+col,bias0);
    if(bias1!=0)atomicAdd(gb+512+col,bias1);
}
'''
    pack=cp.RawKernel(code,'pack');pack.compile();pack.max_dynamic_shared_size_bytes=h*8
    agg=cp.RawKernel(code,'aggregate');agg.compile()
    pa=(wi.data_ptr(),bi.data_ptr(),ids.data_ptr(),masks.data_ptr(),counts.data_ptr())
    aa=(us.data_ptr(),them.data_ptr(),gl.data_ptr(),cl.data_ptr(),np.float32(p['maxact']),ids.data_ptr(),masks.data_ptr(),counts.data_ptr(),gw.data_ptr(),gb.data_ptr())
    def preprocess():pack((tiles,),(256,),pa,shared_mem=h*8)
    def run():
        preprocess();gw.zero_();gb.zero_();agg((tiles,4),(128,),aa)
    run()
    torch.testing.assert_close(gw,refw,atol=3e-7,rtol=3e-4)
    torch.testing.assert_close(gb,refb,atol=3e-7,rtol=3e-4)
    row={'tile':tile,'total_ms':timing(run),'preprocess_ms':timing(preprocess)}
    results['variants'].append(row);print(row,flush=True)
    pathlib.Path('/workspace/results/aggregate-ft.json').write_text(json.dumps(results,indent=2))

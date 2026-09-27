"""Experimental aggregated FT backward; duplicate rows fall back to scatter."""
import os
import sys
sys.path.insert(0, '/workspace/nnue')
import cupy as cp
import torch
from model.modules.feature_transformer import fused_ft_functions as ff
cache={}
@torch.compiler.disable(recursive=False)
def factory(active,width,tile_size=4):
    assert width==1024 and tile_size==4
    if active not in cache:
        cache[active]=build(active)
    return cache[active]
def build(A):
    tile=8
    maxids=2*tile*A
    h=1 << (maxids-1).bit_length()
    code=f'#define T {tile}\n#define A {A}\n#define H {h}\n#define M {maxids}\n'+r'''
extern "C" __global__ void pack(const int* w,const int* b,int* ids,unsigned* masks,int* counts){
    extern __shared__ unsigned mem[];
    int* keys=(int*)mem;
    unsigned* bits=mem+H;
    __shared__ int n, duplicate;
    if(threadIdx.x==0){n=0;duplicate=0;}
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
            if(old==-1||old==id){unsigned oldbits=atomicOr(bits+slot,1u<<row);if(oldbits&(1u<<row))atomicExch(&duplicate,1);break;}
            slot=(slot+1)&(H-1);
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<H;i+=blockDim.x){
        if(keys[i]>=0){int offset=atomicAdd(&n,1);ids[blockIdx.x*M+offset]=keys[i];masks[blockIdx.x*M+offset]=bits[i];}
    }
    __syncthreads();
    if(threadIdx.x==0)counts[blockIdx.x]=duplicate?-n-1:n;
}
extern "C" __global__ void aggregate(const float* us,const float* them,const float* gl,const float* cl,
    float maxact,const int* w,const int* b,const int* ids,const unsigned* masks,const int* counts,float* gw,float* gb){
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
    if(n<0){
        for(int r=0;r<2*T;++r){
            int pos=blockIdx.x*T+r/2;
            const int* row=((r&1)?b:w)+pos*A;
            float v0=g[r][tid],v1=g[r][tid+128];
            for(int k=0;k<A;++k){
                int id=row[k];if(id<0)break;
                if(v0!=0)atomicAdd(gw+id*1024+col,v0);
                if(v1!=0)atomicAdd(gw+id*1024+512+col,v1);
            }
        }
    }
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
    def launch(grid,args):
        B=int(args[-2]); tiles=(B+tile-1)//tile
        assert B%tile==0, 'prototype batch must be divisible by tile'
        ids=torch.empty((tiles,maxids),device='cuda',dtype=torch.int32)
        masks=torch.empty_like(ids)
        counts=torch.empty(tiles,device='cuda',dtype=torch.int32)
        pa=(args[2],args[3],ids.data_ptr(),masks.data_ptr(),counts.data_ptr())
        aa=(args[0],args[1],args[7],args[8],args[6],args[2],args[3],ids.data_ptr(),masks.data_ptr(),counts.data_ptr(),args[9],args[10])
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            pack((tiles,),(256,),pa,shared_mem=h*8)
            agg((tiles,4),(128,),aa)
    return launch

def enable():
    ff.make_fused_double_ft_backward_kernel=factory

if __name__=='__main__':
    torch.set_num_threads(1)
    if '--parity' in sys.argv:
        import quant_experiment
        quant_experiment.enable=enable
        quant_experiment.parity()
    else:
        if os.environ.get('NNUE_AGGREGATE_FT')=='1': enable()
        import runpy
        runpy.run_path('/workspace/nnue/tests/bench_training.py',run_name='__main__')

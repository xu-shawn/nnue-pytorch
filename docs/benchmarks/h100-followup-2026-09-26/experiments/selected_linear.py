"""Experimental selected-bucket FP32 linear kernels; no production dispatch."""
import functools
import cupy as cp
import numpy as np
import torch

_cache={}
@torch.compiler.disable(recursive=False)
def kernels(k,n,threads=128,sparse=False):
    key=(k,n,threads,sparse)
    if key in _cache:return _cache[key]
    warps=threads//32
    code=r'''
#define K @K@
#define N @N@
#define THREADS @THREADS@
#define WARPS (THREADS/32)
#define OUTPUTS ((N+WARPS-1)/WARPS)
#define COLS ((K+THREADS-1)/THREADS)
extern "C" __global__ void selected_forward(const float* x,const float* w,const float* bias,const long long* bucket,float* y){
 int row=blockIdx.x, lane=threadIdx.x%32, warp=threadIdx.x/32;
 int b=bucket[row];
 float acc[OUTPUTS]={0};
 @SPARSE@
 #pragma unroll
 for(int o=0;o<OUTPUTS;++o){
  float a=acc[o];
  #pragma unroll
  for(int d=16;d;d>>=1)a+=__shfl_down_sync(0xffffffff,a,d);
  int col=warp+o*WARPS;
  if(lane==0 && col<N)y[row*N+col]=a+bias[b*N+col];
 }
}
extern "C" __global__ void selected_dx(const float* gy,const float* w,const long long* bucket,float* dx){
 int row=blockIdx.x, b=bucket[row];
 float acc[COLS]={0};
 #pragma unroll
 for(int o=0;o<N;++o){
  float g=gy[row*N+o];
  #pragma unroll
  for(int c=0;c<COLS;++c){int k=threadIdx.x+c*THREADS;if(k<K)acc[c]=fmaf(g,w[(b*N+o)*K+k],acc[c]);}
 }
 #pragma unroll
 for(int c=0;c<COLS;++c){int k=threadIdx.x+c*THREADS;if(k<K)dx[row*K+k]=acc[c];}
}
'''
    dense=r'''
 for(int k=lane;k<K;k+=32){
  float a=x[row*K+k];
  if(a!=0.0f){
   #pragma unroll
   for(int o=0;o<OUTPUTS;++o){int col=warp+o*WARPS;if(col<N)acc[o]=fmaf(a,w[(b*N+col)*K+k],acc[o]);}
  }
 }
'''
    compact=r'''
 __shared__ int nnz;
 __shared__ int indices[K];
 __shared__ float values[K];
 if(threadIdx.x==0)nnz=0;
 __syncthreads();
 for(int k=threadIdx.x;k<K;k+=THREADS){
  float a=x[row*K+k];
  unsigned mask=__ballot_sync(0xffffffff,a!=0.0f);
  int base=0;
  if(lane==0)base=atomicAdd(&nnz,__popc(mask));
  base=__shfl_sync(0xffffffff,base,0);
  if(a!=0.0f){int i=base+__popc(mask&((1u<<lane)-1));indices[i]=k;values[i]=a;}
 }
 __syncthreads();
 for(int i=lane;i<nnz;i+=32){
  int k=indices[i];float a=values[i];
  #pragma unroll
  for(int o=0;o<OUTPUTS;++o){int col=warp+o*WARPS;if(col<N)acc[o]=fmaf(a,w[(b*N+col)*K+k],acc[o]);}
 }
'''
    code=code.replace('@K@',str(k)).replace('@N@',str(n)).replace('@THREADS@',str(threads)).replace('@SPARSE@',compact if sparse else dense)
    mod=cp.RawModule(code=code,options=('--std=c++17',))
    _cache[key]=(mod.get_function('selected_forward'),mod.get_function('selected_dx'))
    return _cache[key]

def forward(x,w,b,indices,n,threads=128,sparse=False):
    f,_=kernels(x.shape[1],n,threads,sparse)
    y=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype)
    f((x.shape[0],),(threads,),(x.data_ptr(),w.data_ptr(),b.data_ptr(),indices.data_ptr(),y.data_ptr()))
    return y

def dx(gy,w,indices,k,threads=128):
    _,f=kernels(k,gy.shape[1],threads,False)
    out=torch.empty((gy.shape[0],k),device=gy.device,dtype=gy.dtype)
    f((gy.shape[0],),(threads,),(gy.data_ptr(),w.data_ptr(),indices.data_ptr(),out.data_ptr()))
    return out

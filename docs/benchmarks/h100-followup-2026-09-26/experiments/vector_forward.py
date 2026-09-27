import cupy as cp
import torch
_cache={}
@torch.compiler.disable(recursive=False)
def make_forward(active,threads=128):
 key=(active,threads)
 if key not in _cache:
  code=r'''
#define THREADS @THREADS@
#define ACTIVE @ACTIVE@
#define SLICES (128/THREADS)
__device__ __forceinline__ float4 add4(float4 a,float4 b){return make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
__device__ __forceinline__ float clamp1(float a,float m){return a<0.0f?0.0f:(a>m?m:a);}
__device__ __forceinline__ float4 mix4(float s,float4 a,float t,float4 b,float m){return make_float4(clamp1(s*a.x+t*b.x,m),clamp1(s*a.y+t*b.y,m),clamp1(s*a.z+t*b.z,m),clamp1(s*a.w+t*b.w,m));}
__device__ __forceinline__ float4 mul4(float4 a,float4 b){return make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
extern "C" __global__ void vector_ft_forward(const float* us,const float* them,const int* wi,const int* bi,const float4* weight,const float4* bias,float maxact,float4* out,float4* clamps,int output_size){
 int row=blockIdx.x,tid=threadIdx.x;
 float u=us[row],t=them[row];
 float4 w0[SLICES],w1[SLICES],b0[SLICES],b1[SLICES];
 #pragma unroll
 for(int s=0;s<SLICES;++s){int col=tid+s*THREADS;w0[s]=bias[col];w1[s]=bias[col+128];b0[s]=w0[s];b1[s]=w1[s];}
 for(int j=0;j<ACTIVE;++j){int idx=wi[row*ACTIVE+j];if(idx<0)break;
  #pragma unroll
  for(int s=0;s<SLICES;++s){int col=tid+s*THREADS;w0[s]=add4(w0[s],weight[idx*256+col]);w1[s]=add4(w1[s],weight[idx*256+col+128]);}
 }
 for(int j=0;j<ACTIVE;++j){int idx=bi[row*ACTIVE+j];if(idx<0)break;
  #pragma unroll
  for(int s=0;s<SLICES;++s){int col=tid+s*THREADS;b0[s]=add4(b0[s],weight[idx*256+col]);b1[s]=add4(b1[s],weight[idx*256+col+128]);}
 }
 #pragma unroll
 for(int s=0;s<SLICES;++s){int col=tid+s*THREADS;
  float4 a=mix4(u,w0[s],t,b0[s],maxact),b=mix4(u,w1[s],t,b1[s],maxact);
  float4 c=mix4(t,w0[s],u,b0[s],maxact),d=mix4(t,w1[s],u,b1[s],maxact);
  out[row*256+col]=mul4(a,b);out[row*256+128+col]=mul4(c,d);
  clamps[row*512+col]=a;clamps[row*512+128+col]=b;clamps[row*512+256+col]=c;clamps[row*512+384+col]=d;
 }
}
'''.replace('@THREADS@',str(threads)).replace('@ACTIVE@',str(active))
  kernel=cp.RawKernel(code,'vector_ft_forward');kernel.compile()
  _cache[key]=lambda grid,args:kernel(grid=grid,block=(threads,),args=args)
 return _cache[key]

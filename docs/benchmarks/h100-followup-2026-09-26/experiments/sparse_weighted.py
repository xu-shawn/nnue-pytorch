"""Experimental weighted L1 SpMM, inspired by Bullet's adjacent-output layout.

Bullet SpMM consumes binary feature indices. These kernels multiply nonzero
quantized activation values and preserve the existing dense STE backward.
"""
import functools
import cupy as cp
import torch


@functools.lru_cache(None)
def kernels(threads=128, mode='ballot', vector=1):
    code = r'''
#define THREADS @THREADS@
#define WARPS (THREADS/32)
#define VEC @VEC@
#define GROUP (32/VEC)
extern "C" __global__ void weighted(const float* x,const float* w,const float* b,const long long* buckets,float* y){
 int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 int bucket=buckets[row];
 float acc=0;
 @BODY@
 __shared__ float partial[WARPS][32];
 partial[warp][lane]=acc;
 __syncthreads();
 if(warp==0){
  float total=b[bucket*32+lane];
  #pragma unroll
  for(int i=0;i<WARPS;i++)total+=partial[i][lane];
  y[row*32+lane]=total;
 }
}
extern "C" __global__ void pack(const float* x,int* ids,float* vals,int* count){
 int row=blockIdx.x,lane=threadIdx.x%32;
 __shared__ int nnz;
 if(threadIdx.x==0)nnz=0;
 __syncthreads();
 for(int k=threadIdx.x;k<1024;k+=THREADS){
  float a=x[row*1024+k];
  unsigned mask=__ballot_sync(0xffffffff,a!=0);
  int base=0;
  if(lane==0)base=atomicAdd(&nnz,__popc(mask));
  base=__shfl_sync(0xffffffff,base,0);
  if(a!=0){int i=base+__popc(mask&((1u<<lane)-1));ids[row*1024+i]=k;vals[row*1024+i]=a;}
 }
 __syncthreads();
 if(threadIdx.x==0)count[row]=nnz;
}
extern "C" __global__ void packed(const int* ids,const float* vals,const int* count,const float* w,const float* b,const long long* buckets,float* y,int batch){
 int t=blockIdx.x*blockDim.x+threadIdx.x,row=t/GROUP,lane=t%GROUP;
 if(row>=batch)return;
 int bucket=buckets[row];
 float acc[VEC]={0};
 for(int i=0;i<count[row];i++){
  int k=ids[row*1024+i];float a=vals[row*1024+i];
  @VECTOR_BODY@
 }
 #pragma unroll
 for(int j=0;j<VEC;j++)y[row*32+lane*VEC+j]=acc[j]+b[bucket*32+lane*VEC+j];
}
'''
    compact = r'''
 __shared__ int nnz,ids[1024];
 __shared__ float vals[1024];
 if(threadIdx.x==0)nnz=0;
 __syncthreads();
 for(int k=threadIdx.x;k<1024;k+=THREADS){
  float a=x[row*1024+k];
  unsigned mask=__ballot_sync(0xffffffff,a!=0);
  int base=0;
  if(lane==0)base=atomicAdd(&nnz,__popc(mask));
  base=__shfl_sync(0xffffffff,base,0);
  if(a!=0){int i=base+__popc(mask&((1u<<lane)-1));ids[i]=k;vals[i]=a;}
 }
 __syncthreads();
 for(int i=warp;i<nnz;i+=WARPS){
  int k=ids[i];float a=vals[i];
  acc=fmaf(a,w[(bucket*1024+k)*32+lane],acc);
 }
'''
    ballot = r'''
 for(int base=warp*32;base<1024;base+=WARPS*32){
  float value=x[row*1024+base+lane];
  unsigned mask=__ballot_sync(0xffffffff,value!=0);
  while(mask){
   int bit=__ffs(mask)-1;
   float a=__shfl_sync(0xffffffff,value,bit);
   acc=fmaf(a,w[(bucket*1024+base+bit)*32+lane],acc);
   mask&=mask-1;
  }
 }
'''
    vb = 'acc[0]=fmaf(a,w[(bucket*1024+k)*32+lane],acc[0]);'
    if vector > 1:
        comps = 'xyzw'[:vector]
        vb = f'float{vector} weights=reinterpret_cast<const float{vector}*>(w)[(bucket*1024+k)*GROUP+lane];'
        vb += ''.join(f'acc[{i}]=fmaf(a,weights.{c},acc[{i}]);' for i,c in enumerate(comps))
    code = code.replace('@THREADS@',str(threads)).replace('@VEC@',str(vector)).replace('@BODY@',compact if mode=='compact' else ballot).replace('@VECTOR_BODY@',vb)
    mod = cp.RawModule(code=code, options=('--std=c++17',))
    return tuple(mod.get_function(n) for n in ['weighted','pack','packed'])


def transpose(weight):
    return weight.reshape(8,32,1024).transpose(1,2).contiguous()


def pack(x,threads=128):
    ids = torch.empty_like(x,dtype=torch.int32)
    vals = torch.empty_like(x)
    counts = torch.empty(len(x),device=x.device,dtype=torch.int32)
    _,fn,_ = kernels(threads)
    stream=cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
    fn((len(x),),(threads,),(x.data_ptr(),ids.data_ptr(),vals.data_ptr(),counts.data_ptr()),stream=stream)
    return ids,vals,counts


def forward(x,wt,b,indices,threads=128,mode='ballot',vector=1,packed=None):
    y=torch.empty((len(x),32),device=x.device)
    direct,_,consume=kernels(threads,mode,vector)
    stream=cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
    if mode=='packed':
        ids,vals,counts=packed if packed is not None else pack(x,threads)
        group=32//vector
        consume(((len(x)*group+threads-1)//threads,),(threads,),(ids.data_ptr(),vals.data_ptr(),counts.data_ptr(),wt.data_ptr(),b.data_ptr(),indices.data_ptr(),y.data_ptr(),len(x)),stream=stream)
    else:
        direct((len(x),),(threads,),(x.data_ptr(),wt.data_ptr(),b.data_ptr(),indices.data_ptr(),y.data_ptr()),stream=stream)
    return y

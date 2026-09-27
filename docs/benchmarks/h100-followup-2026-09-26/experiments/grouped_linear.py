"""Experimental bucket routing and strict-FP32 grouped matmul gradients."""
import cupy as cp
import torch
import triton as tr
import triton.language as tl
_route=None
@torch.compiler.disable(recursive=False)
def route(indices):
 global _route
 if _route is None:
  _route=cp.RawKernel(r'''
extern "C" __global__ void route(const long long* ids,int* rows,int* counts,int batch){
 __shared__ int used[8],base[8];
 int t=threadIdx.x,r=blockIdx.x*blockDim.x+t;
 if(t<8)used[t]=0;
 __syncthreads();
 int b=0,pos=0;
 if(r<batch){b=ids[r];pos=atomicAdd(&used[b],1);}
 __syncthreads();
 if(t<8)base[t]=atomicAdd(&counts[t],used[t]);
 __syncthreads();
 if(r<batch)rows[b*batch+base[b]+pos]=r;
}
''','route')
 rows=torch.empty((8,len(indices)),device=indices.device,dtype=torch.int32)
 counts=torch.zeros(8,device=indices.device,dtype=torch.int32)
 stream=cp.cuda.ExternalStream(torch.cuda.current_stream(indices.device).cuda_stream)
 _route((tr.cdiv(len(indices),256),),(256,),(indices.data_ptr(),rows.data_ptr(),counts.data_ptr(),len(indices)),stream=stream)
 return rows,counts

@tr.jit
def _fwd(X,W,B,R,C,Y,BATCH:tl.constexpr,K:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
 p=tl.program_id(0);bucket=tl.program_id(1);count=tl.load(C+bucket)
 if p*BM<count:
  m=p*BM+tl.arange(0,BM);n=tl.arange(0,BN);ki=tl.arange(0,BK)
  rows=tl.load(R+bucket*BATCH+m,m<count,0)
  acc=tl.full((BM,BN),0,tl.float32)
  for block in range(tl.cdiv(K,BK)):
   k=block*BK+ki
   a=tl.load(X+rows[:,None]*K+k[None,:],(m[:,None]<count)&(k[None,:]<K),0)
   w=tl.load(W+(bucket*N+n[None,:])*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0)
   acc=tl.dot(a,w,acc,input_precision='ieee')
  bias=tl.load(B+bucket*N+n,n<N,0)
  tl.store(Y+rows[:,None]*N+n[None,:],acc+bias[None,:],(m[:,None]<count)&(n[None,:]<N))

@tr.jit
def _dx(G,W,R,C,DX,BATCH:tl.constexpr,K:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
 p=tl.program_id(0);pk=tl.program_id(1);bucket=tl.program_id(2);count=tl.load(C+bucket)
 if p*BM<count:
  m=p*BM+tl.arange(0,BM);k=pk*BK+tl.arange(0,BK);n=tl.arange(0,BN)
  rows=tl.load(R+bucket*BATCH+m,m<count,0)
  g=tl.load(G+rows[:,None]*N+n[None,:],(m[:,None]<count)&(n[None,:]<N),0)
  w=tl.load(W+(bucket*N+n[:,None])*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
  acc=tl.dot(g,w,input_precision='ieee')
  tl.store(DX+rows[:,None]*K+k[None,:],acc,(m[:,None]<count)&(k[None,:]<K))

@tr.jit
def _dw(X,G,R,C,P,Q,BATCH:tl.constexpr,K:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
 pk=tl.program_id(0);b=tl.program_id(1);s=tl.program_id(2)
 k=pk*BK+tl.arange(0,BK);n=tl.arange(0,BN);mi=tl.arange(0,BM)
 count=tl.load(C+b)
 acc=tl.full((BN,BK),0,tl.float32)
 for start in range(s*BM,count,BM*SPLIT):
  m=start+mi;rows=tl.load(R+b*BATCH+m,m<count,0)
  x=tl.load(X+rows[:,None]*K+k[None,:],(m[:,None]<count)&(k[None,:]<K),0)
  g=tl.load(G+rows[None,:]*N+n[:,None],(m[None,:]<count)&(n[:,None]<N),0)
  acc=tl.dot(g,x,acc,input_precision='ieee')
 tl.store(P+((b*SPLIT+s)*N+n[:,None])*K+k[None,:],acc,(n[:,None]<N)&(k[None,:]<K))


@tr.jit
def _db(G,R,C,Q,BATCH:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,BN:tl.constexpr):
 b=tl.program_id(0);s=tl.program_id(1)
 n=tl.arange(0,BN);mi=tl.arange(0,128);count=tl.load(C+b)
 acc=tl.full((BN,),0,tl.float32)
 for start in range(s*128,count,128*SPLIT):
  m=start+mi;rows=tl.load(R+b*BATCH+m,m<count,0)
  g=tl.load(G+rows[:,None]*N+n[None,:],(m[:,None]<count)&(n[None,:]<N),0)
  acc+=tl.sum(g,0)
 tl.store(Q+(b*SPLIT+s)*N+n,acc,n<N)

@tr.jit
def _reduce(P,Q,W,B,K:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,BLOCK:tl.constexpr):
 b=tl.program_id(0);i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
 s=tl.arange(0,SPLIT)
 p=tl.load(P+b*SPLIT*N*K+s[:,None]*N*K+i[None,:],i[None,:]<N*K,0)
 tl.store(W+b*N*K+i,tl.sum(p,0),i<N*K)
 if tl.program_id(1)==0:
  q=tl.load(Q+b*SPLIT*N+s[:,None]*N+i[None,:],i[None,:]<N,0)
  tl.store(B+b*N+i,tl.sum(q,0),i<N)

def forward(x,w,b,rows,counts,n,bm=32,bk=32):
 y=torch.empty((len(x),n),device=x.device,dtype=x.dtype)
 _fwd[(tr.cdiv(len(x),bm),8)](x,w,b,rows,counts,y,len(x),x.shape[1],n,bm,bk,max(16,tr.next_power_of_2(n)),num_warps=4)
 return y

def dx(g,w,rows,counts,k,bm=32,bk=64):
 out=torch.empty((len(g),k),device=g.device,dtype=g.dtype)
 _dx[(tr.cdiv(len(g),bm),tr.cdiv(k,bk),8)](g,w,rows,counts,out,len(g),k,g.shape[1],bm,bk,max(16,tr.next_power_of_2(g.shape[1])),num_warps=4)
 return out

def dw(x,g,rows,counts,split=8,bm=32,bk=32):
 batch,k=x.shape;n=g.shape[1]
 p=torch.empty((8,split,n,k),device=x.device,dtype=x.dtype)
 q=torch.empty((8,split,n),device=x.device,dtype=x.dtype)
 w=torch.empty((8*n,k),device=x.device,dtype=x.dtype);b=torch.empty(8*n,device=x.device,dtype=x.dtype)
 _dw[(tr.cdiv(k,bk),8,split)](x,g,rows,counts,p,q,batch,k,n,split,bm,bk,max(16,tr.next_power_of_2(n)),num_warps=4)
 _db[(8,split)](g,rows,counts,q,batch,n,split,max(16,tr.next_power_of_2(n)),num_warps=4)
 _reduce[(8,tr.cdiv(n*k,128))](p,q,w,b,k,n,split,128,num_warps=4)
 return w,b

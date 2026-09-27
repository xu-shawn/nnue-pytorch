import cupy as cp
from functools import lru_cache

@lru_cache(None)
def make_vector_backward(max_active, l1, tile=4):
    assert l1 % 8 == 0
    code = r'''
extern "C" __global__ void vector_backward(
    const float* us, const float* them, const int* wi, const int* bi,
    const float* weight, const float* bias, float maxact,
    const float* grad, const float* clamped, float* gw, float* gb,
    int batch, int output_size) {
    const int tid=threadIdx.x;
    const int half=L1/2;
    float bias0[4]={0,0,0,0}, bias1[4]={0,0,0,0};
    for (int t=0; t<TILE; ++t) {
        int row=blockIdx.x*TILE+t;
        if (row>=batch) break;
        float u=us[row], v=them[row];
        const float* c=clamped+row*2*L1+tid*4;
        const float* g=grad+row*L1+tid*4;
        float w0[4],w1[4],b0[4],b1[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            float cw0=c[j],cw1=c[half+j],cb0=c[2*half+j],cb1=c[3*half+j];
            float dw0=(cw0==0 || cw0==maxact)?0:g[j]*cw1;
            float dw1=(cw1==0 || cw1==maxact)?0:g[j]*cw0;
            float db0=(cb0==0 || cb0==maxact)?0:g[half+j]*cb1;
            float db1=(cb1==0 || cb1==maxact)?0:g[half+j]*cb0;
            w0[j]=u*dw0+v*db0; w1[j]=u*dw1+v*db1;
            b0[j]=v*dw0+u*db0; b1[j]=v*dw1+u*db1;
            bias0[j]+=w0[j]+b0[j]; bias1[j]+=w1[j]+b1[j];
        }
        float4 wg0=make_float4(w0[0],w0[1],w0[2],w0[3]);
        float4 wg1=make_float4(w1[0],w1[1],w1[2],w1[3]);
        float4 bg0=make_float4(b0[0],b0[1],b0[2],b0[3]);
        float4 bg1=make_float4(b1[0],b1[1],b1[2],b1[3]);
        for(int k=0;k<ACTIVE;++k) {
            int idx=wi[row*ACTIVE+k]; if(idx==-1) break;
            atomicAdd((float4*)(gw+idx*output_size+tid*4),wg0);
            atomicAdd((float4*)(gw+idx*output_size+tid*4+half),wg1);
        }
        for(int k=0;k<ACTIVE;++k) {
            int idx=bi[row*ACTIVE+k]; if(idx==-1) break;
            atomicAdd((float4*)(gw+idx*output_size+tid*4),bg0);
            atomicAdd((float4*)(gw+idx*output_size+tid*4+half),bg1);
        }
    }
    atomicAdd((float4*)(gb+tid*4),make_float4(bias0[0],bias0[1],bias0[2],bias0[3]));
    atomicAdd((float4*)(gb+tid*4+half),make_float4(bias1[0],bias1[1],bias1[2],bias1[3]));
}
'''.replace('ACTIVE',str(max_active)).replace('L1',str(l1)).replace('TILE',str(tile))
    kernel=cp.RawKernel(code,'vector_backward');kernel.compile()
    return lambda grid,args:kernel(grid=grid,block=(l1//8,),args=args)

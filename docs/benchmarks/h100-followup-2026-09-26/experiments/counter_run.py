"""Standalone CuPy replay of real mature-net FT work; also runs on CUDA 12.8+ 5090."""
import argparse
import json
import pathlib
import statistics

import cupy as cp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--probe', action='store_true')
parser.add_argument('--aggregate', action='store_true')
args = parser.parse_args()
if args.probe:
    k = cp.RawKernel('extern "C" __global__ void counter_probe(float* x){x[threadIdx.x]+=1;}', 'counter_probe')
    x = cp.zeros(256, dtype=cp.float32)
    k((1,), (256,), (x,))
    cp.cuda.runtime.deviceSynchronize()
    print('probe complete')
    raise SystemExit

p = np.load('counter-inputs.npz')
us, them, wi, bi, weight, bias, grad = [cp.asarray(p[k]) for k in ['us','them','wi','bi','weight','bias','grad']]
B, A = wi.shape
assert (B,A) == (32768,288)
forward = cp.RawKernel(pathlib.Path('ft-forward.cu').read_text(), 'fused_double_ft_forward')
backward = cp.RawKernel(pathlib.Path('ft-backward.cu').read_text(), 'fused_double_ft_backward')
y = cp.empty((B,1024), dtype=cp.float32)
cl = cp.empty((B,4,512), dtype=cp.float32)
gw, gb = cp.empty_like(weight), cp.empty_like(bias)
fa = (us,them,wi,bi,weight,bias,np.float32(p['maxact']),y,cl,np.int32(1024))
ba = (us,them,wi,bi,weight,bias,np.float32(p['maxact']),grad,cl,gw,gb,np.int32(B),np.int32(1024))
def fwd():forward((B,), (256,), fa)
def bwd():
    gw.fill(0);gb.fill(0)
    backward((B//4,4), (128,), ba)
if args.aggregate:
    code=pathlib.Path('ft-aggregate.cu').read_text()
    pack=cp.RawKernel(code,'ft_pack_features');pack.compile();pack.max_dynamic_shared_size_bytes=65536
    agg=cp.RawKernel(code,'ft_aggregate_backward');agg.compile()
    ids=cp.empty((B//8,16*A),dtype=cp.int32);masks=cp.empty_like(ids);counts=cp.empty(B//8,dtype=cp.int32)
    pa=(wi,bi,ids,masks,counts,np.int32(B))
    aa=(us,them,grad,cl,np.float32(p['maxact']),wi,bi,ids,masks,counts,gw,gb,np.int32(B))
    fwd();bwd();refw=gw.copy();refb=gb.copy()
    def bwd():
        gw.fill(0);gb.fill(0)
        pack((B//8,),(256,),pa,shared_mem=65536)
        agg((B//8,4),(128,),aa)
    bwd();cp.testing.assert_allclose(gw,refw,atol=3e-7,rtol=3e-4);cp.testing.assert_allclose(gb,refb,atol=3e-7,rtol=3e-4)
for _ in range(10):fwd();bwd()
cp.cuda.runtime.deviceSynchronize()
timings={}
for name,fn in [('forward',fwd),('backward_with_zero',bwd)]:
    vals=[]
    for _ in range(5):
        a,b=cp.cuda.Event(),cp.cuda.Event();a.record()
        for _ in range(10):fn()
        b.record();b.synchronize();vals.append(cp.cuda.get_elapsed_time(a,b)/10)
    timings[name]=statistics.median(vals)
info=cp.cuda.runtime.getDeviceProperties(0)
metadata={'gpu':info['name'].decode(),'capability':[info['major'],info['minor']],
          'cupy':cp.__version__,'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),
          'driver':cp.cuda.runtime.driverGetVersion(),'timing_ms':timings,
          'input_shape':[B,A],'note':'Real H100 inputs and launch geometry; cross-architecture diagnostic, not H100 counters.'}
print(json.dumps(metadata,indent=2))
pathlib.Path('counter-aggregate-environment.json' if args.aggregate else 'counter-environment.json').write_text(json.dumps(metadata,indent=2))
cp.cuda.profiler.start()
fwd();bwd()
cp.cuda.runtime.deviceSynchronize()
cp.cuda.profiler.stop()

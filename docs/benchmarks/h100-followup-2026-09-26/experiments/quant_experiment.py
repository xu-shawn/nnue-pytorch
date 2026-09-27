"""Experimental fixed-master-net FT activation quantization fusion."""
import importlib.util
import os
import sys

sys.path.insert(0, '/workspace/nnue')
import cupy as cp
import torch
from model.modules.feature_transformer import fused_ft_functions as ff
from model.quantize import QuantizationManager

original_factory = ff.make_fused_double_ft_forward_kernel
original_quant = QuantizationManager.fake_quantize_ft_act
cache = {}

@torch.compiler.disable(recursive=False)
def factory(active, width):
    assert width == 1024 and torch.cuda.get_device_capability() == (9, 0)
    if (active, width) not in cache:
        wrapper = original_factory(active, width)
        raw = next(c.cell_contents for c in wrapper.__closure__ if isinstance(c.cell_contents, cp.RawKernel))
        code = raw.code
        for pov in ('w', 'b'):
            code = code.replace(f'= l0_{pov}0 * l0_{pov}1;', f'= floorf(__fadd_rn(__fmul_rn(__fmul_rn(l0_{pov}0,l0_{pov}1),128.0f),1.0e-5f))*(1.0f/128.0f);')
        kernel = cp.RawKernel(code, 'fused_double_ft_forward')
        kernel.compile()
        def run(grid, args):
            kernel(grid, (256,), args)
        cache[active, width] = run
    return cache[active, width]

def quant_identity(self, value):
    assert self.config.hidden_quantized_one == 128 and self.l0_correction_factor == 1
    return value

def enable():
    ff.make_fused_double_ft_forward_kernel = factory
    QuantizationManager.fake_quantize_ft_act = quant_identity

def parity():
    from model import NNUE
    spec = importlib.util.spec_from_file_location('bench', '/workspace/nnue/tests/bench_training.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    model = NNUE(config=bench.threats_config(True, True), max_epoch=4500, num_batches_per_epoch=1024).cuda()
    model.load_state_dict(torch.load('/workspace/mature-init.ckpt', map_location='cuda', weights_only=False)['state_dict'])
    batch = torch.load('/workspace/cache-skip0/batch-131072-rank0-of4.pt', map_location='cuda', weights_only=True)[0]
    torch.manual_seed(42)
    loss = model.compute_loss(batch, 0)
    loss.backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    enable()
    torch.manual_seed(42)
    fused_loss = model.compute_loss(batch, 0)
    fused_loss.backward()
    torch.testing.assert_close(fused_loss, loss, atol=0, rtol=0)
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=3e-7, rtol=3e-4, msg=n)
    print('PARITY PASS', len(expected), 'parameter gradients; loss', loss.item(), flush=True)

if __name__ == '__main__':
    torch.set_num_threads(1)
    if '--parity' in sys.argv:
        parity()
    else:
        if os.environ.get('NNUE_FUSED_FT_QUANT') == '1':
            enable()
        import runpy
        runpy.run_path('/workspace/nnue/tests/bench_training.py', run_name='__main__')

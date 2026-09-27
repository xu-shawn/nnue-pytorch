"""Isolated experiment launcher; all switches recorded in benchmark environment."""
import functools, os, runpy, sys
sys.path.insert(0, '/workspace/nnue')
if 'NNUE_GIL_INTERVAL' in os.environ:
    sys.setswitchinterval(float(os.environ['NNUE_GIL_INTERVAL']))
import trainer.engine as engine
if 'NNUE_BUCKET_MB' in os.environ:
    class TunedDDP(engine.DDP):
        def __init__(self, *args, **kwargs):
            kwargs['bucket_cap_mb'] = int(os.environ['NNUE_BUCKET_MB'])
            super().__init__(*args, **kwargs)
    engine.DDP = TunedDDP
if os.environ.get('NNUE_KEEP_GRADS') == '1':
    import torch
    original = torch.optim.Optimizer.zero_grad
    def zero(self, set_to_none=True):
        return original(self, set_to_none=False)
    torch.optim.Optimizer.zero_grad = zero
if 'NNUE_COLSPLIT' in os.environ:
    import pathlib, torch
    from model.modules.feature_transformer import fused_ft_kernel as fk, fused_ft_functions as ff
    parts = int(os.environ['NNUE_COLSPLIT'])
    tile = int(os.environ.get('NNUE_TILE', '4'))
    source = pathlib.Path(fk.__file__).read_text().replace('min(l1_half, 1024)', f'l1_half // {parts}')
    pos = source.index('void fused_double_ft_backward(')
    head, body = source[:pos], source[pos:]
    body = body.replace('const uint32_t tid = threadIdx.x;', 'const uint32_t tid = threadIdx.x + blockIdx.y * blockDim.x;')
    body = body.replace('str(num_threads)', 'str(l1_half)')
    ns = fk.__dict__.copy()
    exec(head + body, ns)
    split_cache = {}
    @torch.compiler.disable(recursive=False)
    def make_split(active, l1):
        key = (active, l1)
        if key not in split_cache:
            kernel = ns['make_fused_double_ft_backward_kernel'](active, l1, tile_size=tile)
            split_cache[key] = lambda grid, args: kernel(grid=(grid[0], parts), args=args)
        return split_cache[key]
    ff.make_fused_double_ft_backward_kernel = make_split
    ff.BACKWARD_TILE_SIZE = tile
if 'NNUE_VECTOR_TILE' in os.environ:
    from model.modules.feature_transformer import fused_ft_functions as ff
    from vector_kernel import make_vector_backward
    tile = int(os.environ['NNUE_VECTOR_TILE'])
    ff.BACKWARD_TILE_SIZE = tile
    ff.make_fused_double_ft_backward_kernel = lambda active, l1: make_vector_backward(active, l1, tile)
runpy.run_path('/workspace/nnue/tests/bench_training.py', run_name='__main__')

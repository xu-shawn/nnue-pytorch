"""Gradient-norm clipping that leaves the gradients untouched when they are within the limit.

`torch.nn.utils.clip_grad_norm_` multiplies every gradient by the clamped coefficient even when it
is 1 (it avoids reading the coefficient on the host). For the feature-transformer gradients that is
a full pass over several hundred MB per step. Here large CUDA gradients are scaled by a kernel that
reads the coefficient on the device and returns at once when it is 1; the result is the same.
"""

import torch

try:
    import cupy as cp
except (ImportError, OSError, RuntimeError):
    cp = None

_MIN_ELEMENTS = 1 << 20
_GRID, _THREADS = 2048, 256
_kernel = None


def _scale_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(
            r"""
extern "C" __global__ void scale_if_clipped(float* __restrict__ g, const float* __restrict__ coef,
                                            const unsigned long long n) {
    const float c = *coef;
    if (c == 1.0f) return;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        g[i] *= c;
}
""",
            "scale_if_clipped",
        )
    return _kernel


@torch.compiler.disable
def _scale_if_clipped(grad: torch.Tensor, coef: torch.Tensor) -> None:
    with cp.cuda.Device(grad.device.index):
        _scale_kernel()(
            (_GRID,),
            (_THREADS,),
            (grad.data_ptr(), coef.data_ptr(), cp.uint64(grad.numel())),
            stream=cp.cuda.ExternalStream(
                torch.cuda.current_stream(grad.device).cuda_stream
            ),
        )


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
    """Same result as torch.nn.utils.clip_grad_norm_(parameters, max_norm)."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    total_norm = torch.nn.utils.get_total_norm(grads, 2.0)
    coef = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
    for g in grads:
        c = coef.to(g.device)
        if (
            cp is not None
            and g.is_cuda
            and torch.version.hip is None
            and g.dtype == torch.float32
            and c.dtype == torch.float32
            and g.is_contiguous()
            and g.numel() >= _MIN_ELEMENTS
        ):
            _scale_if_clipped(g, c.reshape(1))
        else:
            g.mul_(c)
    return total_norm

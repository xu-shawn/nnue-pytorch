"""Tiled backward of the fused double feature transformer.

The tiled scatter (tiled_ft_backward) sums the gradients of the two raw
accumulators (white, black) of every position, each (B, L1), into the
weight rows. Those gradients come from grad_l0 and the forward's clamped
values (the prologue: the per-position part of the fused function's
backward). By default the scatter computes them while loading its tiles
(fused), so they are never written to memory; only positions with
us / them other than (1, 0) / (0, 1) get their rows written by the
reference formula first. ``backward(fused=False)`` writes all rows with
the reference formula and is kept as the reference for tests.
"""

from __future__ import annotations

import numpy as np
import torch

from . import tiled_ft_backward as _tb
from .cupy_utils import compile_kernel, cupy_stream

_PROLOGUE_SRC = r"""
// Reference prologue for one (position, column pair): the accumulator gradients from grad_l0 and the
// clamped values [w0 w1 b0 b1] (H each) the fused forward saved.
__device__ __forceinline__ void pro_elem(const int b, const int i,
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ clamped_out, const float max_act,
    float* __restrict__ grad_w, float* __restrict__ grad_b)
{
    const float u = us[b], th = them[b];
    const size_t r = (size_t)b * $L1, c = (size_t)b * 4 * $H;
    const float cw0 = clamped_out[c + 0 * $H + i], cw1 = clamped_out[c + 1 * $H + i];
    const float cb0 = clamped_out[c + 2 * $H + i], cb1 = clamped_out[c + 3 * $H + i];
    const float g = grad_l0[r + i], gh = grad_l0[r + $H + i];
    const float dw0 = (cw0 == 0.0f || cw0 == max_act) ? 0.0f : g * cw1;
    const float dw1 = (cw1 == 0.0f || cw1 == max_act) ? 0.0f : g * cw0;
    const float db0 = (cb0 == 0.0f || cb0 == max_act) ? 0.0f : gh * cb1;
    const float db1 = (cb1 == 0.0f || cb1 == max_act) ? 0.0f : gh * cb0;
    grad_w[r + i] = u * dw0 + th * db0;
    grad_w[r + $H + i] = u * dw1 + th * db1;
    grad_b[r + i] = th * dw0 + u * db0;
    grad_b[r + $H + i] = th * dw1 + u * db1;
}

__device__ __forceinline__ bool binary_pos(const float u, const float th)
{
    return (u == 1.0f && th == 0.0f) || (u == 0.0f && th == 1.0f);
}

// All positions (the unfused path).
extern "C" __global__ void tiled_ft_prologue(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ clamped_out, const float max_act,
    float* __restrict__ grad_w, float* __restrict__ grad_b, const int batch_size)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)batch_size * $H) return;
    pro_elem((int)(t / $H), (int)(t % $H), us, them, grad_l0, clamped_out, max_act, grad_w, grad_b);
}

// Fused path: per list in tile order, its id, the half of grad_l0 and the clamped pair it reads (white:
// 0 when us == 1), and whether its position needs the reference rows (us / them not (1, 0) or (0, 1)).
extern "C" __global__ void tiled_ft_pro_codes(const int* __restrict__ order, const float* __restrict__ us,
    const float* __restrict__ them, unsigned* __restrict__ code, const int n_lists, const int batch_size)
{
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n_lists) return;
    const int lid = order[li];
    const bool white = lid < batch_size;
    const int b = white ? lid : lid - batch_size;
    const float u = us[b], th = them[b];
    code[li] = (unsigned)lid | ((white == (u == 1.0f)) ? 0u : (1u << 30)) | (binary_pos(u, th) ? 0u : (1u << 31));
}

// Fused path: one warp per position writes the reference rows of the non-binary positions only.
extern "C" __global__ void tiled_ft_pro_ref(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ clamped_out, const float max_act,
    float* __restrict__ grad_w, float* __restrict__ grad_b, const int batch_size)
{
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= batch_size) return;
    const int b = (int)w;
    if (binary_pos(us[b], them[b])) return;
    for (int i = threadIdx.x & 31; i < $H; i += 32)
        pro_elem(b, i, us, them, grad_l0, clamped_out, max_act, grad_w, grad_b);
}
"""


def _kernel(name: str, l1_size: int):
    h = l1_size // 2
    src = _PROLOGUE_SRC.replace("$H", str(h)).replace("$L1", str(l1_size))
    return compile_kernel((name, l1_size), src, name)


def lookup_backward_prep(white_indices):
    return _tb.lookup_prep(white_indices) if _tb.TILED_BACKWARD_ENABLED else None


@torch.compiler.disable(recursive=True)
def backward(
    us,
    them,
    white_indices,
    black_indices,
    grad_l0,
    clamped_out,
    num_inputs,
    max_ft_activation,
    l1_size,
    prep=None,
    fused: bool = True,
):
    """(grad_weight, grad_bias) of the fused double FT, or None when the tiled backward does not apply.
    ``fused=False`` materializes the accumulator gradients first (reference path, same results)."""
    if not _tb.TILED_BACKWARD_ENABLED or not _tb.applicable(l1_size, num_inputs):
        return None
    batch_size = white_indices.shape[0]
    max_act = np.float32(max_ft_activation)
    grad_w = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
    grad_b = torch.empty_like(grad_w)
    args = (
        us.data_ptr(),
        them.data_ptr(),
        grad_l0.data_ptr(),
        clamped_out.data_ptr(),
        max_act,
        grad_w.data_ptr(),
        grad_b.data_ptr(),
        np.int32(batch_size),
    )
    if not fused:
        n = batch_size * (l1_size // 2)
        with cupy_stream(us.device):
            _kernel("tiled_ft_prologue", l1_size)(
                grid=((n + 255) // 256,), block=(256,), args=args
            )
        return _tb.tiled_backward(
            white_indices, black_indices, grad_w, grad_b, num_inputs, l1_size, prep=prep
        )
    if prep is None:
        prep = _tb.fallback_prep(white_indices, black_indices)
    n_lists = 2 * batch_size
    assert n_lists < 1 << 30  # list ids share their code word with two flag bits
    code = torch.empty(n_lists, dtype=torch.int32, device=us.device)
    with cupy_stream(us.device):
        _kernel("tiled_ft_pro_codes", l1_size)(
            grid=((n_lists + 255) // 256,),
            block=(256,),
            args=(
                prep.order.data_ptr(),
                us.data_ptr(),
                them.data_ptr(),
                code.data_ptr(),
                np.int32(n_lists),
                np.int32(batch_size),
            ),
        )
        _kernel("tiled_ft_pro_ref", l1_size)(
            grid=((batch_size * 32 + 255) // 256,), block=(256,), args=args
        )
    return _tb.tiled_backward(
        white_indices,
        black_indices,
        grad_w,
        grad_b,
        num_inputs,
        l1_size,
        prep=prep,
        pro=(grad_l0, clamped_out, code, max_act),
    )

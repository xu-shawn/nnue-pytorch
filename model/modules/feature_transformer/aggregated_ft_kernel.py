"""H100 master-net FT backward with tile-local feature aggregation.

Pack the union of features in eight positions with a position/perspective mask.
Each column block sums those contributions before issuing global FP32 atomics.
A repeated index within one perspective requires multiplicity, so those tiles
fall back to individual scatters. No sparsity approximation is used.
"""

from functools import lru_cache

import cupy as cp
import numpy as np
import torch


@lru_cache(maxsize=None)
def _kernels(active: int):
    A = active
    tile = 8
    maxids = 2 * tile * A
    h = 1 << (maxids - 1).bit_length()
    code = (
        f"#define T {tile}\n#define A {A}\n#define H {h}\n#define M {maxids}\n"
        + r"""
extern "C" __global__ void ft_pack_features(const int *w, const int *b, int *ids, unsigned *masks,
                                            int *counts, int batch_size) {
    extern __shared__ unsigned mem[];
    int *keys = (int *)mem;
    unsigned *bits = mem + H;
    __shared__ int n, duplicate;
    if (threadIdx.x == 0) {
        n = 0;
        duplicate = 0;
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        keys[i] = -1;
        bits[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int row = i / A, k = i % A;
        int pos = blockIdx.x * T + row / 2;
        if (pos >= batch_size)
            continue;
        int id = (row & 1) ? b[pos * A + k] : w[pos * A + k];
        if (id < 0)
            continue;
        unsigned slot = ((unsigned)id * 2654435761u) & (H - 1);
        while (true) {
            int old = atomicCAS(keys + slot, -1, id);
            if (old == -1 || old == id) {
                unsigned oldbits = atomicOr(bits + slot, 1u << row);
                if (oldbits & (1u << row))
                    atomicExch(&duplicate, 1);
                break;
            }
            slot = (slot + 1) & (H - 1);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        if (keys[i] >= 0) {
            int offset = atomicAdd(&n, 1);
            ids[blockIdx.x * M + offset] = keys[i];
            masks[blockIdx.x * M + offset] = bits[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        counts[blockIdx.x] = duplicate ? -n - 1 : n;
}
extern "C" __global__ void ft_aggregate_backward(const float *us, const float *them,
                                                 const float *gl, const float *cl, float maxact,
                                                 const int *w, const int *b, const int *ids,
                                                 const unsigned *masks, const int *counts,
                                                 float *gw, float *gb, int batch_size) {
    __shared__ float g[2 * T][256];
    int tid = threadIdx.x, col = tid + 128 * blockIdx.y;
    float bias0 = 0, bias1 = 0;
    for (int t = 0; t < T; ++t) {
        int row = blockIdx.x * T + t;
        if (row >= batch_size)
            break;
        float w0 = cl[row * 2048 + col], w1 = cl[row * 2048 + 512 + col];
        float b0 = cl[row * 2048 + 1024 + col], b1 = cl[row * 2048 + 1536 + col];
        float d0 = gl[row * 1024 + col], d1 = gl[row * 1024 + 512 + col];
        float dw0 = (w0 == 0 || w0 == maxact) ? 0 : d0 * w1;
        float dw1 = (w1 == 0 || w1 == maxact) ? 0 : d0 * w0;
        float db0 = (b0 == 0 || b0 == maxact) ? 0 : d1 * b1;
        float db1 = (b1 == 0 || b1 == maxact) ? 0 : d1 * b0;
        float u = us[row], v = them[row];
        float gw0 = u * dw0 + v * db0, gw1 = u * dw1 + v * db1;
        float gb0 = v * dw0 + u * db0, gb1 = v * dw1 + u * db1;
        g[2 * t][tid] = gw0;
        g[2 * t][tid + 128] = gw1;
        g[2 * t + 1][tid] = gb0;
        g[2 * t + 1][tid + 128] = gb1;
        bias0 += gw0 + gb0;
        bias1 += gw1 + gb1;
    }
    __syncthreads();
    int n = counts[blockIdx.x];
    if (n < 0) {
        for (int r = 0; r < 2 * T; ++r) {
            int pos = blockIdx.x * T + r / 2;
            if (pos >= batch_size)
                break;
            const int *row = ((r & 1) ? b : w) + pos * A;
            float v0 = g[r][tid], v1 = g[r][tid + 128];
            for (int k = 0; k < A; ++k) {
                int id = row[k];
                if (id < 0)
                    break;
                if (v0 != 0)
                    atomicAdd(gw + id * 1024 + col, v0);
                if (v1 != 0)
                    atomicAdd(gw + id * 1024 + 512 + col, v1);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        int id = ids[blockIdx.x * M + i];
        unsigned mask = masks[blockIdx.x * M + i];
        float v0 = 0, v1 = 0;
        while (mask) {
            int r = __ffs(mask) - 1;
            mask &= mask - 1;
            v0 += g[r][tid];
            v1 += g[r][tid + 128];
        }
        if (v0 != 0)
            atomicAdd(gw + id * 1024 + col, v0);
        if (v1 != 0)
            atomicAdd(gw + id * 1024 + 512 + col, v1);
    }
    if (bias0 != 0)
        atomicAdd(gb + col, bias0);
    if (bias1 != 0)
        atomicAdd(gb + 512 + col, bias1);
}
"""
    )
    pack = cp.RawKernel(code, "ft_pack_features")
    pack.compile()
    pack.max_dynamic_shared_size_bytes = h * 8
    aggregate = cp.RawKernel(code, "ft_aggregate_backward")
    aggregate.compile()
    return pack, aggregate, maxids, h * 8


@torch.compiler.disable
def aggregated_ft_backward(
    us, them, white, black, grad, clamped, grad_weight, grad_bias, maxact
):
    """Accumulate into zeroed output gradients on the current PyTorch stream."""
    batch_size, active = white.shape
    tiles = (batch_size + 7) // 8
    with cp.cuda.Device(us.device.index):
        pack, aggregate, capacity, shared_bytes = _kernels(active)
        ids = torch.empty((tiles, capacity), device=us.device, dtype=torch.int32)
        masks = torch.empty_like(ids)
        counts = torch.empty(tiles, device=us.device, dtype=torch.int32)
        pack_args = (
            white.data_ptr(),
            black.data_ptr(),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            np.int32(batch_size),
        )
        backward_args = (
            us.data_ptr(),
            them.data_ptr(),
            grad.data_ptr(),
            clamped.data_ptr(),
            np.float32(maxact),
            white.data_ptr(),
            black.data_ptr(),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            grad_weight.data_ptr(),
            grad_bias.data_ptr(),
            np.int32(batch_size),
        )
        with cp.cuda.ExternalStream(torch.cuda.current_stream(us.device).cuda_stream):
            pack((tiles,), (256,), pack_args, shared_mem=shared_bytes)
            aggregate((tiles, 4), (128,), backward_args)

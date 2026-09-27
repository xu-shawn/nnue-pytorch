"""Selected-bucket L1 matmul for the H100 master-net workload.

Only the requested one of eight 1024→32 layers is evaluated, with NNZ-compacted
forward. A GPU row map allows tiled FP32 backward matmuls without sorting/copying
the large activation tensor or reading bucket sizes on the host. Split reductions
keep weight gradients parallel even when bucket populations differ. Input
gradients remain dense: fake quantization uses STE, so a quantized zero can have
a nonzero derivative.
"""

import cupy as cp
import torch
import triton as tr
import triton.language as tl

_route = None
_sparse_forward_kernel = None


def _sparse_forward(x, weight, bias, indices):
    """Compact NNZ in shared memory and coalesce loads over adjacent outputs.

    Four warps split each row's nonzeros. The transpose is included in the
    measured cost; parameter storage and dense STE input gradients stay intact.
    """
    global _sparse_forward_kernel
    if _sparse_forward_kernel is None:
        _sparse_forward_kernel = cp.RawKernel(
            r"""
extern "C" __global__ void sparse_l1_forward(
    const float* x, const float* weight, const float* bias,
    const long long* buckets, float* output
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int bucket = buckets[row];
    __shared__ int nnz, indices[1024];
    __shared__ float values[1024], partial[4][32];
    if (threadIdx.x == 0) nnz = 0;
    __syncthreads();

    for (int k = threadIdx.x; k < 1024; k += 128) {
        const float value = x[row * 1024 + k];
        const unsigned mask = __ballot_sync(0xffffffff, value != 0.0f);
        int base = 0;
        if (lane == 0) base = atomicAdd(&nnz, __popc(mask));
        base = __shfl_sync(0xffffffff, base, 0);
        if (value != 0.0f) {
            const int slot = base + __popc(mask & ((1u << lane) - 1));
            indices[slot] = k;
            values[slot] = value;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int i = warp; i < nnz; i += 4) {
        const int k = indices[i];
        acc = fmaf(values[i], weight[(bucket * 1024 + k) * 32 + lane], acc);
    }
    partial[warp][lane] = acc;
    __syncthreads();
    if (warp == 0) {
        float total = bias[bucket * 32 + lane];
        #pragma unroll
        for (int i = 0; i < 4; ++i) total += partial[i][lane];
        output[row * 32 + lane] = total;
    }
}
""",
            "sparse_l1_forward",
        )
    transposed = weight.reshape(8, 32, 1024).transpose(1, 2).contiguous()
    output = torch.empty((len(x), 32), device=x.device, dtype=x.dtype)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
    _sparse_forward_kernel(
        (len(x),),
        (128,),
        (x.data_ptr(), transposed.data_ptr(), bias.data_ptr(), indices.data_ptr(), output.data_ptr()),
        stream=stream,
    )
    return output


@torch.compiler.disable(recursive=False)
def _route_rows(indices):
    global _route
    if _route is None:
        _route = cp.RawKernel(
            r"""
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
""",
            "route",
        )
    rows = torch.empty((8, len(indices)), device=indices.device, dtype=torch.int32)
    counts = torch.zeros(8, device=indices.device, dtype=torch.int32)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream(indices.device).cuda_stream)
    _route(
        (tr.cdiv(len(indices), 256),),
        (256,),
        (indices.data_ptr(), rows.data_ptr(), counts.data_ptr(), len(indices)),
        stream=stream,
    )
    return rows, counts


@tr.jit
def _bucketed_input_gradient(
    G,
    W,
    R,
    C,
    DX,
    BATCH: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    p = tl.program_id(0)
    pk = tl.program_id(1)
    bucket = tl.program_id(2)
    count = tl.load(C + bucket)
    if p * BM < count:
        m = p * BM + tl.arange(0, BM)
        k = pk * BK + tl.arange(0, BK)
        n = tl.arange(0, BN)
        rows = tl.load(R + bucket * BATCH + m, m < count, 0)
        g = tl.load(
            G + rows[:, None] * N + n[None, :],
            (m[:, None] < count) & (n[None, :] < N),
            0,
        )
        w = tl.load(
            W + (bucket * N + n[:, None]) * K + k[None, :],
            (n[:, None] < N) & (k[None, :] < K),
            0,
        )
        acc = tl.dot(g, w, input_precision="ieee")
        tl.store(
            DX + rows[:, None] * K + k[None, :],
            acc,
            (m[:, None] < count) & (k[None, :] < K),
        )


@tr.jit
def _bucketed_weight_gradient(
    X,
    G,
    R,
    C,
    P,
    BATCH: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    pk = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    k = pk * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    mi = tl.arange(0, BM)
    count = tl.load(C + b)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(s * BM, count, BM * SPLIT):
        m = start + mi
        rows = tl.load(R + b * BATCH + m, m < count, 0)
        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            (m[:, None] < count) & (k[None, :] < K),
            0,
        )
        g = tl.load(
            G + rows[None, :] * N + n[:, None],
            (m[None, :] < count) & (n[:, None] < N),
            0,
        )
        acc = tl.dot(g, x, acc, input_precision="ieee")
    tl.store(
        P + ((b * SPLIT + s) * N + n[:, None]) * K + k[None, :],
        acc,
        (n[:, None] < N) & (k[None, :] < K),
    )


@tr.jit
def _bucketed_bias_gradient(
    G,
    R,
    C,
    Q,
    BATCH: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    n = tl.arange(0, BN)
    mi = tl.arange(0, 128)
    count = tl.load(C + b)
    acc = tl.full((BN,), 0, tl.float32)
    for start in range(s * 128, count, 128 * SPLIT):
        m = start + mi
        rows = tl.load(R + b * BATCH + m, m < count, 0)
        g = tl.load(
            G + rows[:, None] * N + n[None, :],
            (m[:, None] < count) & (n[None, :] < N),
            0,
        )
        acc += tl.sum(g, 0)
    tl.store(Q + (b * SPLIT + s) * N + n, acc, n < N)


@tr.jit
def _sum_gradient_partials(
    P,
    Q,
    W,
    B,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLIT)
    p = tl.load(P + b * SPLIT * N * K + s[:, None] * N * K + i[None, :], i[None, :] < N * K, 0)
    tl.store(W + b * N * K + i, tl.sum(p, 0), i < N * K)
    if tl.program_id(1) == 0:
        q = tl.load(Q + b * SPLIT * N + s[:, None] * N + i[None, :], i[None, :] < N, 0)
        tl.store(B + b * N + i, tl.sum(q, 0), i < N)


def _input_gradient(g, w, rows, counts, k, bm=32, bk=64):
    out = torch.empty((len(g), k), device=g.device, dtype=g.dtype)
    _bucketed_input_gradient[(tr.cdiv(len(g), bm), tr.cdiv(k, bk), 8)](
        g,
        w,
        rows,
        counts,
        out,
        len(g),
        k,
        g.shape[1],
        bm,
        bk,
        max(16, tr.next_power_of_2(g.shape[1])),
        num_warps=4,
    )
    return out


def _weight_and_bias_gradient(x, g, rows, counts, split=8, bm=32, bk=32):
    batch, k = x.shape
    n = g.shape[1]
    p = torch.empty((8, split, n, k), device=x.device, dtype=x.dtype)
    q = torch.empty((8, split, n), device=x.device, dtype=x.dtype)
    w = torch.empty((8 * n, k), device=x.device, dtype=x.dtype)
    b = torch.empty(8 * n, device=x.device, dtype=x.dtype)
    _bucketed_weight_gradient[(tr.cdiv(k, bk), 8, split)](
        x,
        g,
        rows,
        counts,
        p,
        batch,
        k,
        n,
        split,
        bm,
        bk,
        max(16, tr.next_power_of_2(n)),
        num_warps=4,
    )
    _bucketed_bias_gradient[(8, split)](
        g, rows, counts, q, batch, n, split, max(16, tr.next_power_of_2(n)), num_warps=4
    )
    _sum_gradient_partials[(8, tr.cdiv(n * k, 128))](p, q, w, b, k, n, split, 128, num_warps=4)
    return w, b


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, indices):
        x = x.contiguous()
        weight = weight.contiguous()
        indices = indices.flatten().to(torch.int64).contiguous()
        rows, counts = _route_rows(indices)
        ctx.save_for_backward(x, weight, rows, counts)
        return _sparse_forward(x, weight, bias.contiguous(), indices)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rows, counts = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # Quantized zero activations still need dense STE input gradients.
        grad_input = _input_gradient(grad_output, weight, rows, counts, 1024, 64, 64)
        grad_weight, grad_bias = _weight_and_bias_gradient(x, grad_output, rows, counts, 16, 32, 32)
        return grad_input, grad_weight, grad_bias, None


@torch.compiler.disable
def grouped_l1(x, weight, bias, indices):
    """FP32 1024→32 linear for eight buckets; indices must be in [0, 8).

    Bucket counts stay on the GPU. Parameters and output row order are identical
    to a dense 1024→256 linear followed by selection. First-order training only,
    as with the custom feature transformer. The caller handles other shapes.
    """
    return _GroupedLinear.apply(x, weight, bias, indices)

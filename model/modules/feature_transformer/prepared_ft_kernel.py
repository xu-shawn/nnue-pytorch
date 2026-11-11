"""Feature-transformer weight table built in one pass, and the forward pass over the int16 table.

The feature transformer reads `weight + tiled virtual weight`, fake-quantized, for every feature
block. Built with tensor operations that is a dozen passes over a table of several hundred MB per
step. One kernel writes the table instead, and only the rows the batch uses: fake-quantized
weights as int16 (the forward pass sums them exactly as integers, which equals the float32 sum of
the rounded weights bit for bit while half the memory is read), or the plain float32 sum when
fake quantization is off.
"""

from functools import cache

import cupy as cp
import torch

from .fused_ft_kernel import _num_threads

_MARK_THREADS = 32
_PREPARE_THREADS = 256


@cache
def _prepare_kernels(active: int, width: int):
    code = (
        f"#define A {active}\n#define K {width}\n"
        + r"""
extern "C" __global__ void ft_mark_rows(
    const int* __restrict__ white, const int* __restrict__ black,
    unsigned char* __restrict__ used
) {
    // Every writer stores the same value, so the races are benign. Reading first keeps the
    // rows most positions share from being written over and over.
    const int* const wr = white + (size_t)blockIdx.x * A;
    const int* const br = black + (size_t)blockIdx.x * A;
    for (int k = threadIdx.x; k < A; k += blockDim.x) {
        const int wi = wr[k], bi = br[k];
        if (wi < 0 && bi < 0) break;  // padding is at the end of a row
        if (wi >= 0 && !used[wi]) used[wi] = 1;
        if (bi >= 0 && !used[bi]) used[bi] = 1;
    }
}

extern "C" __global__ void ft_prepare_q(
    const float* __restrict__ w, const float* __restrict__ v, short* __restrict__ out,
    const unsigned char* __restrict__ used, int* __restrict__ flag,
    const unsigned n, const unsigned period, const float scale
) {
    // Four elements per thread: a 16-byte load and an 8-byte store.
    const unsigned i = 4u * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || !used[i / K]) return;
    float4 a = *reinterpret_cast<const float4*>(w + i);
    if (period) {
        const float4 b = *reinterpret_cast<const float4*>(v + i % period);
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    const float t0 = rintf(a.x * scale), t1 = rintf(a.y * scale);
    const float t2 = rintf(a.z * scale), t3 = rintf(a.w * scale);
    // Out of the int16 range or not finite: poison the forward pass instead of training on.
    if (!(fabsf(t0) <= 32767.0f && fabsf(t1) <= 32767.0f && fabsf(t2) <= 32767.0f && fabsf(t3) <= 32767.0f))
        *flag = 1;
    const unsigned long long packed =
        (unsigned long long)(unsigned short)(short)(int)t0
        | ((unsigned long long)(unsigned short)(short)(int)t1 << 16)
        | ((unsigned long long)(unsigned short)(short)(int)t2 << 32)
        | ((unsigned long long)(unsigned short)(short)(int)t3 << 48);
    *reinterpret_cast<unsigned long long*>(out + i) = packed;
}

extern "C" __global__ void ft_prepare_f(
    const float* __restrict__ w, const float* __restrict__ v, float* __restrict__ out,
    const unsigned char* __restrict__ used, const unsigned n, const unsigned period
) {
    const unsigned i = 4u * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || !used[i / K]) return;
    float4 a = *reinterpret_cast<const float4*>(w + i);
    if (period) {
        const float4 b = *reinterpret_cast<const float4*>(v + i % period);
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    *reinterpret_cast<float4*>(out + i) = a;
}
"""
    )
    kernels = [
        cp.RawKernel(code, name)
        for name in ("ft_mark_rows", "ft_prepare_q", "ft_prepare_f")
    ]
    for kernel in kernels:
        kernel.compile()
    return kernels


@torch.compiler.disable
def prepare_ft_table(parts, white, black, width: int, quantize: bool, scale: float):
    """Build the table for `parts` = [(weight, virtual or None, period rows)], stacked by rows.

    Returns (table, flag): int16 (fake-quantized, scale applied) or float32, and for int16 a
    one-element flag set when some weight did not fit. Only the rows named in `white` and
    `black` are written.
    """
    device = white.device
    rows = sum(w.shape[0] for w, _, _ in parts)
    table = torch.empty(
        (rows, width), dtype=torch.int16 if quantize else torch.float32, device=device
    )
    flag = torch.zeros(1, dtype=torch.int32, device=device) if quantize else None
    used = torch.zeros(rows, dtype=torch.uint8, device=device)
    mark, kq, kf = _prepare_kernels(white.shape[1], width)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
    with cp.cuda.Device(device.index):
        mark(
            (white.shape[0],),
            (_MARK_THREADS,),
            (white.data_ptr(), black.data_ptr(), used.data_ptr()),
            stream=stream,
        )
        offset = 0
        for w, v, period in parts:
            n = w.numel()
            grid = ((n // 4 + _PREPARE_THREADS - 1) // _PREPARE_THREADS,)
            out = table.data_ptr() + offset * width * table.element_size()
            head = (w.data_ptr(), v.data_ptr() if v is not None else 0, out)
            tail = (cp.uint32(n), cp.uint32(period * width if v is not None else 0))
            mask = used.data_ptr() + offset
            if quantize:
                args = (*head, mask, flag.data_ptr(), *tail, cp.float32(scale))
                kq(grid, (_PREPARE_THREADS,), args, stream=stream)
            else:
                kf(grid, (_PREPARE_THREADS,), (*head, mask, *tail), stream=stream)
            offset += w.shape[0]
    return table, flag


@cache
def _forward_q_kernel(active: int, width: int):
    half = width // 2
    threads = _num_threads(half // 2, 128)
    code = (
        f"#define A {active}\n#define K {width}\n#define HALF {half}\n"
        f"#define NT {threads}\n#define S {half // 2 // threads}\n"
        + r"""
extern "C" __global__ void ft_forward_q(
    const float* __restrict__ us, const float* __restrict__ them,
    const int* __restrict__ white, const int* __restrict__ black,
    const short* __restrict__ table, const float* __restrict__ bias,
    const int* __restrict__ flag, const float scale, const float max_act,
    float* __restrict__ l0, float* __restrict__ clamped
) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const int* const wr = white + (size_t)row * A;
    const int* const br = black + (size_t)row * A;
    // Thread tid owns columns 2 * (s * NT + tid) + {0, 1} of each half: one 32-bit load reads
    // both (measured on H100: 128 threads with two such loads per half beat one column a load).
    int w0[2 * S], w1[2 * S], b0[2 * S], b1[2 * S];
    #pragma unroll
    for (unsigned j = 0; j < 2 * S; ++j) { w0[j] = 0; w1[j] = 0; b0[j] = 0; b1[j] = 0; }

    // Integer sums are exact, so the order of the rows does not matter.
    for (int k = 0; k < A; ++k) {
        const int wi = wr[k], bi = br[k];
        if (wi == -1 && bi == -1) break;
        #pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            const unsigned i = 2 * (s * NT + tid);
            if (wi >= 0) {
                const short* const r = table + (size_t)(unsigned)wi * K + i;
                const unsigned x0 = __ldg(reinterpret_cast<const unsigned*>(r));
                const unsigned x1 = __ldg(reinterpret_cast<const unsigned*>(r + HALF));
                w0[2 * s] += (short)x0; w0[2 * s + 1] += (short)(x0 >> 16);
                w1[2 * s] += (short)x1; w1[2 * s + 1] += (short)(x1 >> 16);
            }
            if (bi >= 0) {
                const short* const r = table + (size_t)(unsigned)bi * K + i;
                const unsigned x0 = __ldg(reinterpret_cast<const unsigned*>(r));
                const unsigned x1 = __ldg(reinterpret_cast<const unsigned*>(r + HALF));
                b0[2 * s] += (short)x0; b0[2 * s + 1] += (short)(x0 >> 16);
                b1[2 * s] += (short)x1; b1[2 * s + 1] += (short)(x1 >> 16);
            }
        }
    }

    const float u = __ldg(&us[row]), v = __ldg(&them[row]);
    const bool poisoned = __ldg(flag) != 0;
    #pragma unroll
    for (unsigned j = 0; j < 2 * S; ++j) {
        const unsigned i = 2 * ((j / 2) * NT + tid) + j % 2;
        // The fake-quantized bias is rint(bias * scale) / scale; all terms share the grid.
        const float q0 = rintf(__ldg(&bias[i]) * scale), q1 = rintf(__ldg(&bias[i + HALF]) * scale);
        float fw0 = ((float)w0[j] + q0) / scale, fw1 = ((float)w1[j] + q1) / scale;
        float fb0 = ((float)b0[j] + q0) / scale, fb1 = ((float)b1[j] + q1) / scale;
        if (poisoned) { fw0 = fw1 = fb0 = fb1 = __int_as_float(0x7fc00000); }

        float l0_w0 = u * fw0 + v * fb0;
        float l0_w1 = u * fw1 + v * fb1;
        float l0_b0 = u * fb0 + v * fw0;
        float l0_b1 = u * fb1 + v * fw1;

        if (l0_w0 < 0.0f) l0_w0 = 0.0f; else if (l0_w0 > max_act) l0_w0 = max_act;
        if (l0_w1 < 0.0f) l0_w1 = 0.0f; else if (l0_w1 > max_act) l0_w1 = max_act;
        if (l0_b0 < 0.0f) l0_b0 = 0.0f; else if (l0_b0 > max_act) l0_b0 = max_act;
        if (l0_b1 < 0.0f) l0_b1 = 0.0f; else if (l0_b1 > max_act) l0_b1 = max_act;

        l0[(size_t)row * K + i] = l0_w0 * l0_w1;
        l0[(size_t)row * K + HALF + i] = l0_b0 * l0_b1;
        const size_t base = (size_t)row * 4 * HALF;
        clamped[base + i] = l0_w0;
        clamped[base + HALF + i] = l0_w1;
        clamped[base + 2 * HALF + i] = l0_b0;
        clamped[base + 3 * HALF + i] = l0_b1;
    }
}
"""
    )
    kernel = cp.RawKernel(code, "ft_forward_q")
    kernel.compile()
    return kernel, threads


@torch.compiler.disable
def ft_forward_q(
    us, them, white, black, table, bias, flag, scale, max_act, l0, clamped
):
    batch, active = white.shape
    width = table.shape[1]
    kernel, threads = _forward_q_kernel(active, width)
    with cp.cuda.Device(us.device.index):
        kernel(
            (batch,),
            (threads,),
            (
                us.data_ptr(),
                them.data_ptr(),
                white.data_ptr(),
                black.data_ptr(),
                table.data_ptr(),
                bias.data_ptr(),
                flag.data_ptr(),
                cp.float32(scale),
                cp.float32(max_act),
                l0.data_ptr(),
                clamped.data_ptr(),
            ),
            stream=cp.cuda.ExternalStream(
                torch.cuda.current_stream(us.device).cuda_stream
            ),
        )

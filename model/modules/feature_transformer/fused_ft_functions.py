import numpy as np
import torch
from torch import autograd

_HAS_CUPY_KERNELS = False
try:
    from .aggregated_ft_kernel import aggregated_ft_backward
    from .fused_ft_kernel import (
        BACKWARD_TILE_SIZE,
        make_fused_double_ft_backward_kernel,
        make_fused_double_ft_forward_kernel,
    )
    from .prepared_ft_kernel import ft_forward_q, prepare_ft_table

    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    BACKWARD_TILE_SIZE = 1


def _use_aggregated_backward(
    l1_size, batch_size, max_active_features, device, *, mode="auto"
):
    """Select measured defaults; allow other NVIDIA GPUs to be benchmarked."""
    if mode not in ("auto", "scatter", "aggregated"):
        raise ValueError(f"mode must be auto, scatter, or aggregated, got {mode!r}")
    if mode == "scatter" or torch.version.hip is not None:
        return False
    if not (
        512 <= l1_size <= 4096 and l1_size % 128 == 0 and 0 < max_active_features <= 288
    ):
        return False
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        return False
    if mode == "aggregated":
        return True
    # H100 retains its existing threshold. RTX 4090 and A100 measurements show
    # a benefit from 4096 positions; smaller batches keep direct scatter.
    return (
        (capability == (9, 0) and batch_size >= 1024)
        or (capability == (8, 9) and batch_size >= 4096)
        or (capability == (8, 0) and batch_size >= 4096)
    )


def _aggregated_tile_size(l1_size, batch_size, max_active_features, capability):
    """Position-tile size for the aggregation kernel, per measured device.

    A larger tile removes more repeated feature atomics on the production net.
    The tile must stay at or below 16 positions: rows are tracked in a 32-bit
    mask (two rows per position, one per perspective).
    """
    if l1_size == 1024 and max_active_features == 288:
        if capability == (8, 9) and batch_size >= 32768:
            return 14
        if capability == (8, 0) and batch_size >= 8192:
            return 12
        if capability == (9, 0) and batch_size >= 8192:
            return 14
    return 8


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        max_ft_activation,
        l1_size,
    ):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert (
            us.device
            == them.device
            == white_indices.device
            == black_indices.device
            == weight.device
            == bias.device
        )

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous()
        assert weight.is_contiguous() and bias.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        clamped_out = torch.empty(
            batch_size, 4, l1_half, dtype=torch.float32, device=us.device
        )

        output_size = bias.shape[0]
        kernel = make_fused_double_ft_forward_kernel(max_active_features, l1_size)
        kernel(
            grid=(batch_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                l0_.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(output_size),
            ),
        )

        ctx.save_for_backward(
            us, them, white_indices, black_indices, weight, bias, clamped_out
        )
        return l0_

    @staticmethod
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, weight, bias, clamped_out = (
            ctx.saved_tensors
        )
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()

        output_size = bias.shape[0]

        grad_weight = torch.zeros(
            weight.shape[0], output_size, dtype=torch.float32, device=us.device
        )
        grad_bias = torch.zeros(output_size, dtype=torch.float32, device=us.device)
        _ft_backward(
            us,
            them,
            white_indices,
            black_indices,
            grad_l0,
            clamped_out,
            grad_weight,
            grad_bias,
            max_ft_activation,
            l1_size,
        )
        return None, None, None, None, grad_weight, grad_bias, None, None


def _ft_backward(
    us,
    them,
    white_indices,
    black_indices,
    grad_l0,
    clamped_out,
    grad_weight,
    grad_bias,
    max_ft_activation,
    l1_size,
):
    """Accumulate the weight-table and bias gradients into zeroed buffers."""
    batch_size = white_indices.shape[0]
    max_active_features = white_indices.shape[1]
    output_size = grad_bias.shape[0]

    # Feature extractors guarantee unique IDs within each perspective.
    # Keep direct scatter for unsupported shapes/devices and small batches.
    if _use_aggregated_backward(l1_size, batch_size, max_active_features, us.device):
        tile_size = _aggregated_tile_size(
            l1_size,
            batch_size,
            max_active_features,
            torch.cuda.get_device_capability(us.device),
        )
        aggregated_ft_backward(
            us,
            them,
            white_indices,
            black_indices,
            grad_l0,
            clamped_out,
            grad_weight,
            grad_bias,
            max_ft_activation,
            tile_size=tile_size,
        )
        return

    kernel = make_fused_double_ft_backward_kernel(max_active_features, l1_size)
    grid_size = (batch_size + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE
    kernel(
        grid=(grid_size,),
        args=(
            us.data_ptr(),
            them.data_ptr(),
            white_indices.data_ptr(),
            black_indices.data_ptr(),
            0,  # weight (unused)
            0,  # bias (unused)
            np.float32(max_ft_activation),
            grad_l0.data_ptr(),
            clamped_out.data_ptr(),
            grad_weight.data_ptr(),
            grad_bias.data_ptr(),
            np.int32(batch_size),
            np.int32(output_size),
        ),
    )


class PreparedDoubleFtFunction(autograd.Function):
    """Double feature transform straight from the feature blocks' parameters.

    Inputs after the fixed arguments are, per feature block, its weight and (when `periods` gives
    a nonzero period for it) its virtual weight, tiled over the block's rows with that period.
    The merged table (fake-quantized to `scale` when `quantize`) is built in one pass and is not
    part of the graph: fake quantization is a straight-through estimator, so the table gradient
    is the weight gradient of every block, and summed over the tiles it is the virtual one.
    """

    @staticmethod
    def forward(
        ctx,
        us,
        them,
        white_indices,
        black_indices,
        bias,
        max_ft_activation,
        l1_size,
        quantize,
        scale,
        periods,
        *params,
    ):
        assert us.is_cuda and us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert white_indices.shape == black_indices.shape and white_indices.ndim == 2
        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous()
        assert bias.is_contiguous() and bias.shape == (l1_size,)

        parts = []
        it = iter(params)
        for period in periods:
            weight = next(it)
            virtual = next(it) if period else None
            assert weight.dtype == torch.float32 and weight.is_contiguous()
            assert weight.shape[1] == l1_size
            if virtual is not None:
                assert virtual.is_contiguous() and virtual.shape == (period, l1_size)
                assert weight.shape[0] % period == 0
            parts.append((weight, virtual, period))

        batch_size = white_indices.shape[0]
        l1_half = l1_size // 2
        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        clamped_out = torch.empty(
            batch_size, 4, l1_half, dtype=torch.float32, device=us.device
        )
        table, flag = prepare_ft_table(
            parts, white_indices, black_indices, l1_size, bool(quantize), float(scale)
        )
        if quantize:
            ft_forward_q(
                us,
                them,
                white_indices,
                black_indices,
                table,
                bias,
                flag,
                float(scale),
                float(max_ft_activation),
                l0_,
                clamped_out,
            )
        else:
            kernel = make_fused_double_ft_forward_kernel(
                white_indices.shape[1], l1_size
            )
            kernel(
                grid=(batch_size,),
                args=(
                    us.data_ptr(),
                    them.data_ptr(),
                    white_indices.data_ptr(),
                    black_indices.data_ptr(),
                    table.data_ptr(),
                    bias.data_ptr(),
                    np.float32(max_ft_activation),
                    l0_.data_ptr(),
                    clamped_out.data_ptr(),
                    np.int32(l1_size),
                ),
            )

        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)
        ctx.layout = [(w.shape[0], period) for w, _, period in parts]
        ctx.save_for_backward(us, them, white_indices, black_indices, clamped_out)
        return l0_

    @staticmethod
    @autograd.function.once_differentiable
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, clamped_out = ctx.saved_tensors
        l1_size = ctx.l1_size
        rows = sum(n for n, _ in ctx.layout)
        grad_table = torch.zeros(rows, l1_size, dtype=torch.float32, device=us.device)
        grad_bias = torch.zeros(l1_size, dtype=torch.float32, device=us.device)
        _ft_backward(
            us,
            them,
            white_indices,
            black_indices,
            grad_l0.contiguous(),
            clamped_out,
            grad_table,
            grad_bias,
            ctx.max_ft_activation,
            l1_size,
        )
        grads = []
        offset = 0
        for n, period in ctx.layout:
            block = grad_table.narrow(0, offset, n)
            grads.append(block)
            if period:
                grads.append(block.view(n // period, period, l1_size).sum(0))
            offset += n
        return (None, None, None, None, grad_bias, None, None, None, None, None, *grads)

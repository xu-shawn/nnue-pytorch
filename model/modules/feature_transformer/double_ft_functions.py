import torch

from .fused_ft_functions import (
    _HAS_CUPY_KERNELS,
    FusedDoubleFtFunction,
    PreparedDoubleFtFunction,
)
from .sparse_linear_functions import SparseLinearFunction


def prepared_double_feature_transform_available(
    parts, us: torch.Tensor, l1_size: int
) -> bool:
    """Whether the feature transform can run straight from the feature blocks' parameters
    (`parts` = their table_parts()): NVIDIA GPUs with the CuPy kernels, float32 parameters."""
    if not (_HAS_CUPY_KERNELS and us.is_cuda and torch.version.hip is None):
        return False
    if l1_size % 4 != 0 or any(p is None for p in parts):
        return False
    rows = sum(w.shape[0] for w, _, _ in parts)
    return rows * l1_size < 2**31 and all(
        w.is_cuda and w.dtype == torch.float32 for w, _, _ in parts
    )


def prepared_double_feature_transform(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    parts,
    bias: torch.Tensor,
    max_ft_activation: float,
    l1_size: int,
    quantize: bool,
    scale: float,
) -> torch.Tensor:
    params = []
    for weight, virtual, period in parts:
        params.append(weight)
        if period:
            params.append(virtual)
    return PreparedDoubleFtFunction.apply(
        us,
        them,
        white_indices,
        black_indices,
        bias,
        max_ft_activation,
        l1_size,
        quantize,
        scale,
        tuple(period for _, _, period in parts),
        *params,
    )


def double_feature_transform(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    max_ft_activation: float,
    l1_size: int,
    backend: str = "auto",
) -> torch.Tensor:
    # Resolve backend
    cupy_available = _HAS_CUPY_KERNELS
    all_cuda = (
        us.is_cuda
        and them.is_cuda
        and white_indices.is_cuda
        and black_indices.is_cuda
        and weight.is_cuda
        and bias.is_cuda
    )
    cuda_capable = cupy_available and all_cuda

    if backend == "auto":
        impl = "fused" if cuda_capable else "torch"
    else:
        impl = backend

    if impl == "fused":
        if not cupy_available:
            raise RuntimeError(
                "Fused double FT backend requested, but CuPy kernels are not available."
            )
        if not all_cuda:
            raise RuntimeError(
                "Fused double FT backend requested, but not all tensors/parameters are on CUDA."
            )
        return FusedDoubleFtFunction.apply(
            us,
            them,
            white_indices,
            black_indices,
            weight,
            bias,
            max_ft_activation,
            l1_size,
        )
    elif impl in ("sparse", "torch"):
        if impl == "sparse":
            if not cupy_available:
                raise RuntimeError(
                    "Sparse backend requested, but CuPy kernels are not available."
                )
            if not all_cuda:
                raise RuntimeError(
                    "Sparse backend requested, but not all tensors/parameters are on CUDA."
                )

        assert l1_size % 2 == 0

        w = SparseLinearFunction.apply(white_indices, weight, bias, backend=impl)
        b = SparseLinearFunction.apply(black_indices, weight, bias, backend=impl)

        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        # do not fake quantize sum of (quantized) weights
        l0_ = torch.clamp(l0_, 0.0, max_ft_activation)

        l0_s = torch.split(l0_, l1_size // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_ = torch.cat(l0_s1, dim=1)

        return l0_
    else:
        raise ValueError(f"Invalid double FT implementation mode: {backend}")

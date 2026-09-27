"""Custom FT kernels must honor the caller's CUDA stream."""

import pytest
import torch

from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
                    reason="CUDA and CuPy required")
@pytest.mark.parametrize("backend", ["fused", "sparse"])
@pytest.mark.parametrize("width,batch", [(128, 17), (1152, 1025)])
def test_ft_on_nondefault_stream(backend, width, batch):
    torch.manual_seed(17)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep input creation pending while the host queues the custom kernel.
        torch.cuda._sleep(2_000_000)
        us = torch.randint(0, 2, (batch, 1), device="cuda").float()
        them = 1 - us
        white = torch.randint(0, 64, (batch, 17), device="cuda", dtype=torch.int32)
        black = torch.randint(0, 64, (batch, 17), device="cuda", dtype=torch.int32)
        white[:, 9:] = -1
        black[:, 13:] = -1
        weight = (torch.randn(64, width, device="cuda") / 16).requires_grad_()
        bias = torch.full((width,), 0.5, device="cuda", requires_grad=True)
        dy = torch.randn(batch, width, device="cuda")
        args = (us, them, white, black, weight, bias, 255 / 256, width)
        actual = double_feature_transform(*args, backend)
        # The experimental H100 training path stores forward weights in half.
        # Match that rounding while retaining a FP32 autograd reference.
        compact = (backend == "fused" and batch >= 1024 and width % 128 == 0
                   and torch.cuda.get_device_capability() == (9, 0))
        if compact:
            from model.modules.feature_transformer.sparse_linear_functions import (
                SparseLinearFunction,
            )

            reference_weight = weight.detach().half().float().requires_grad_()
            w = SparseLinearFunction.apply(white, reference_weight, bias, "torch")
            b = SparseLinearFunction.apply(black, reference_weight, bias, "torch")
            pre = us * torch.cat((w, b), dim=1) + them * torch.cat((b, w), dim=1)
            clamped = pre.clamp(0, 255 / 256)
            # Fused FT defines zero derivative at both clamp boundaries;
            # torch.clamp passes it through. Half rounding can hit them exactly.
            interior = (pre > 0) & (pre < 255 / 256)
            strict_clamp = clamped.detach() + torch.where(interior, pre - pre.detach(), 0.0)
            w0, w1, b0, b1 = strict_clamp.chunk(4, dim=1)
            expected = torch.cat((w0 * w1, b0 * b1), dim=1)
        else:
            reference_weight = weight
            expected = double_feature_transform(*args, "torch")
        actual_grads = torch.autograd.grad(actual, (weight, bias), dy)
        expected_grads = torch.autograd.grad(expected, (reference_weight, bias), dy)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=4e-4)
    stream.synchronize()

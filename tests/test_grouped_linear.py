import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.stacked_linear import FactorizedStackedLinear, grouped_l1
from model.quantize import QuantizationConfig, QuantizationManager


@pytest.mark.skipif(not torch.cuda.is_available() or grouped_l1 is None, reason="CUDA, CuPy and Triton required")
@pytest.mark.parametrize("batch", [1, 17, 257])
@pytest.mark.parametrize("concentrated", [False, True])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_grouped_forward_and_all_gradients(batch, concentrated, quantize, sparse):
    torch.manual_seed(123)
    layer = FactorizedStackedLinear(1024, 32, 8, QuantizationManager(QuantizationConfig()), "ls_l1").cuda()
    # Strided all-zero, dense and 75%-zero inputs; uneven/empty buckets and partial tiles.
    storage = torch.randn(batch, 1024, 2, device="cuda")
    zero_fraction = {1: 1.0, 17: 0.0, 257: 0.75}[batch]
    storage[..., 0].masked_fill_(torch.rand(batch, 1024, device="cuda") < zero_fraction, 0)
    x = storage[..., 0].detach().requires_grad_()
    indices = torch.randint(0, 8, (batch, 1), device="cuda", dtype=torch.int32)
    if concentrated:
        indices.fill_(7)
    upstream = torch.randn(batch, 32, device="cuda")
    parameters = tuple(layer.parameters())

    # Both the CuPy router and Triton matmuls must respect the caller's stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference = layer(x, indices, quantize)
        expected = torch.autograd.grad(reference, (x, *parameters), upstream)
        layer.use_grouped = True
        layer.use_sparse = sparse
        actual = layer(x, indices, quantize)
        gradients = torch.autograd.grad(actual, (x, *parameters), upstream)
    torch.cuda.current_stream().wait_stream(stream)

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    for grad, ref_grad in zip(gradients, expected):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
    # A zero activation does not permit dropping its straight-through gradient.
    if zero_fraction:
        assert torch.count_nonzero(gradients[0][x == 0]) > 0


def test_grouped_cpu_fallback():
    layer = FactorizedStackedLinear(1024, 32, 8, QuantizationManager(QuantizationConfig()), "ls_l1")
    x = torch.randn(3, 1024)
    indices = torch.tensor([[0], [4], [7]])
    expected = layer(x, indices)
    layer.use_grouped = True
    layer.use_sparse = True
    torch.testing.assert_close(layer(x, indices), expected)

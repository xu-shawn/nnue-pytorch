import os
import sys
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules import stacked_linear
from model.modules.stacked_linear import FactorizedStackedLinear, grouped_l1
from model.quantize import QuantizationConfig, QuantizationManager


CUDA_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None
OPTIMIZED_AVAILABLE = (
    CUDA_AVAILABLE
    and grouped_l1 is not None
    and torch.cuda.get_device_capability()[0] >= 8
)


def _reference(layer, x, indices, quantize=False):
    weight = layer.linear.weight.reshape(layer.count, layer.out_features, layer.in_features)
    weight = weight + layer.factorized_linear.weight
    bias = layer.linear.bias.reshape(layer.count, layer.out_features) + layer.factorized_linear.bias
    if quantize:
        weight = layer.quantization.fake_quantize_weights(weight, "ls_l1_weight")
        bias = layer.quantization.fake_quantize_weights(bias, "ls_l1_bias")
    indices = indices.flatten().long()
    return torch.bmm(weight[indices], x.unsqueeze(-1)).squeeze(-1) + bias[indices]


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
@pytest.mark.parametrize("batch", [1, 17, 257])
@pytest.mark.parametrize("concentrated", [False, True])
@pytest.mark.parametrize("quantize", [False, True])
def test_grouped_forward_and_all_gradients(batch, concentrated, quantize, monkeypatch):
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
    optimized = Mock(wraps=grouped_l1)
    monkeypatch.setattr(stacked_linear, "grouped_l1", optimized)

    # Both the CuPy router and Triton matmuls must respect the caller's stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference = _reference(layer, x, indices, quantize)
        expected = torch.autograd.grad(reference, (x, *parameters), upstream)
        actual = layer(x, indices, quantize)
        gradients = torch.autograd.grad(actual, (x, *parameters), upstream)
    torch.cuda.current_stream().wait_stream(stream)
    optimized.assert_called_once()

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    for grad, ref_grad in zip(gradients, expected):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
    # A zero activation does not permit dropping its straight-through gradient.
    if zero_fraction:
        assert torch.count_nonzero(gradients[0][x == 0]) > 0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="NVIDIA CUDA required")),
])
@pytest.mark.parametrize("case", ["dependencies", "width", "outputs", "buckets", "dtype", "empty", "autocast"])
def test_grouped_fallback(device, case, monkeypatch):
    width = 512 if case == "width" else 1024
    outputs = 16 if case == "outputs" else 32
    buckets = 4 if case == "buckets" else 8
    dtype = torch.float64 if case == "dtype" else torch.float32
    batch = 0 if case == "empty" else 3
    layer = FactorizedStackedLinear(width, outputs, buckets, QuantizationManager(QuantizationConfig()), "ls_l1")
    layer = layer.to(device=device, dtype=dtype)
    x = torch.randn(batch, width, device=device, dtype=dtype, requires_grad=True)
    indices = torch.arange(batch, device=device).reshape(-1, 1) % buckets
    optimized = Mock(side_effect=AssertionError("Unsupported inputs must use the PyTorch fallback"))
    monkeypatch.setattr(stacked_linear, "grouped_l1", None if case == "dependencies" else optimized)

    with torch.autocast(device_type=device, enabled=case == "autocast"):
        actual = layer(x, indices)
        # Use the original all-buckets operation for matching autocast semantics.
        if case == "autocast":
            weight = layer.linear.weight + layer.factorized_linear.weight.repeat(buckets, 1)
            bias = layer.linear.bias + layer.factorized_linear.bias.repeat(buckets)
            expected = layer.select_output(torch.nn.functional.linear(x, weight, bias), indices)
        else:
            expected = _reference(layer, x, indices)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    parameters = (x, *layer.parameters())
    gradients = torch.autograd.grad(actual.sum(), parameters)
    reference_gradients = torch.autograd.grad(expected.sum(), parameters)
    for grad, ref_grad in zip(gradients, reference_gradients):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
    optimized.assert_not_called()


def test_grouped_cpu_fallback_compiles():
    layer = FactorizedStackedLinear(1024, 32, 8, QuantizationManager(QuantizationConfig()), "ls_l1")
    x = torch.randn(3, 1024, requires_grad=True)
    indices = torch.tensor([[0], [4], [7]])
    compiled = torch.compile(layer, backend="aot_eager")
    actual = compiled(x, indices, True)
    expected = _reference(layer, x, indices, True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    parameters = (x, *layer.parameters())
    gradients = torch.autograd.grad(actual.sum(), parameters)
    reference_gradients = torch.autograd.grad(expected.sum(), parameters)
    for grad, ref_grad in zip(gradients, reference_gradients):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)

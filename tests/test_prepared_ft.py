"""The one-pass weight table path against the tensor-operation path of the feature transformer."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.modules import ComposedFeatureTransformer
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS
from model.modules.features import get_feature_cls
from model.quantize import QuantizationManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _HAS_CUPY_KERNELS
    or torch.version.hip is not None,
    reason="NVIDIA GPU and CuPy required",
)


def _transformer(width):
    quantization = QuantizationManager(ModelConfig().quantize_config)
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"), width, quantization
    ).cuda()
    with torch.no_grad():
        # Wider than the int8 export range: the forward pass rounds but does not clamp.
        for f in ft.features:
            f.weight.uniform_(-0.6, 0.6)
        ft.features[2].virtual_weight.uniform_(-0.2, 0.2)
        ft.bias.uniform_(-0.5, 0.5)
    return ft


def _batch(ft, batch, seed):
    g = torch.Generator().manual_seed(seed)
    active = ft.MAX_ACTIVE_FEATURES
    white = torch.full((batch, active), -1, dtype=torch.int32)
    black = torch.full((batch, active), -1, dtype=torch.int32)
    for indices in (white, black):
        for row in range(batch):
            n = int(torch.randint(0, 97, (1,), generator=g))
            if row == batch - 1:
                n = active  # a full row: no padding
            # Unique within the row, padding at the end, as the data loader guarantees.
            ids = torch.randint(0, ft.NUM_INPUTS, (2 * n,), generator=g).unique()
            ids = ids[torch.randperm(len(ids), generator=g)][:n]
            indices[row, : len(ids)] = ids.int()
    us = torch.randint(0, 2, (batch, 1), generator=g).float()
    return us.cuda(), (1 - us).cuda(), white.cuda(), black.cuda()


@pytest.mark.parametrize("width", [1024, 1152, 256])
@pytest.mark.parametrize("batch", [1, 37, 2048])
@pytest.mark.parametrize("quantize", [True, False])
def test_prepared_matches_merged(width, batch, quantize):
    torch.manual_seed(5)
    ft = _transformer(width)
    us, them, white, black = _batch(ft, batch, 11 + batch)
    probe = torch.randn(batch, width, device="cuda")
    params = [ft.bias] + [p for f in ft.features for p in f.parameters()]

    results = []
    for backend in ("fused", "auto"):
        out = ft(us, them, white, black, False, quantize, backend=backend)
        grads = torch.autograd.grad((out * probe).sum(), params)
        results.append((out.detach(), grads))
    (ref, ref_grads), (out, grads) = results

    # Every weight is on the 1/256 grid when quantized, so the sums are exact either way;
    # unquantized, both paths add the same float32 rows in the same order.
    assert torch.equal(out, ref)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=2e-4, atol=2e-5)


def test_prepared_poisons_unrepresentable_weights():
    ft = _transformer(256)
    us, them, white, black = _batch(ft, 8, 3)
    # A weight of the first feature block in a row the batch uses, and one it does not.
    used = int(white[white < ft.features[0].NUM_INPUTS].max())
    unused = next(
        r
        for r in range(ft.features[0].NUM_INPUTS)
        if not ((white == r) | (black == r)).any()
    )
    weight = ft.features[0].weight
    with torch.no_grad():
        weight[unused, 7] = 200.0  # never read: no effect
        assert torch.isfinite(ft(us, them, white, black, False, True)).all()
        weight[used, 7] = 200.0  # outside the int16 table at scale 256
        assert torch.isnan(ft(us, them, white, black, False, True)).all()
        weight[used, 7] = float("nan")
        assert torch.isnan(ft(us, them, white, black, False, True)).all()
        weight[used, 7] = 0.25
        assert torch.isfinite(ft(us, them, white, black, False, True)).all()

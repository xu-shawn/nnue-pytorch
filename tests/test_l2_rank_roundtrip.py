"""Serialization round trip of the factorised l1 (--l2-rank)."""

import io

import pytest
import torch

import model as M
from model.config import ModelConfig
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = "Full_Threats+PP_3Wide+K32Q2^"


@pytest.mark.parametrize("l2,rank", [(64, 32), (32, 16)])
def test_l2_rank_roundtrip(l2, rank):
    torch.manual_seed(0)
    config = ModelConfig()
    config.L2 = l2
    config.L2R = rank
    model = M.NNUEModel(FEATURES, config)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    model.clip_weights(include_input=True)
    first = bytes(NNUEWriter(model, verbose=False).buf)
    reader = NNUEReader(io.BytesIO(first), FEATURES, config)
    assert reader.model.layer_stacks.l1mix is not None
    second = bytes(NNUEWriter(reader.model, verbose=False).buf)
    assert first == second

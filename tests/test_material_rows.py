"""Material rows (--material-rows): serialization round trip and hash."""

import io

import pytest
import torch

import model as M
from model.config import ModelConfig
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = "Full_Threats+PP_3Wide+K32Q2P2^"


def test_material_rows_roundtrip():
    torch.manual_seed(0)
    config = ModelConfig()
    config.material_rows = True
    model = M.NNUEModel(FEATURES, config)
    with torch.no_grad():
        model.mat_ft.uniform_(-0.5, 0.5)
    first = bytes(NNUEWriter(model, verbose=False).buf)
    reader = NNUEReader(io.BytesIO(first), FEATURES, config)
    # the kings' rows are folded into the bias on export
    assert reader.model.mat_ft[10:].abs().sum() == 0
    assert first == bytes(NNUEWriter(reader.model, verbose=False).buf)
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(first), FEATURES, ModelConfig())

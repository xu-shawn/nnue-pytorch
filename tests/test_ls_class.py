"""Material-class layer stacks (--ls-class-max): serialization round trip and hash."""

import io

import pytest
import torch

import model as M
from model.config import ModelConfig
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = "Full_Threats+PP_3Wide+K32Q2^"


def test_ls_class_roundtrip_and_hash():
    torch.manual_seed(0)
    config = ModelConfig()
    config.ls_class_max = 12
    model = M.NNUEModel(FEATURES, config)
    first = bytes(NNUEWriter(model, verbose=False).buf)
    reader = NNUEReader(io.BytesIO(first), FEATURES, config)
    assert first == bytes(NNUEWriter(reader.model, verbose=False).buf)
    # nets with queen-bucket stacks must not load as material-class nets and vice versa
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(first), FEATURES, ModelConfig())

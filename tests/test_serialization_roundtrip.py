import io
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter


def test_random_weights_roundtrip():
    torch.manual_seed(42)

    config = ModelConfig()
    model = NNUEModel(
        feature_name=DEFAULT_FEATURES,
        config=config,
    )
    # Initialize random weights covering full dynamic range and hitting clipping thresholds
    init_random_weights(model)

    # 1. Export initial model
    writer1 = NNUEWriter(model, description="Test Random Net", verbose=False)
    bytes1 = bytes(writer1.buf)
    assert len(bytes1) > 0

    # 2. Read back into model2
    reader1 = NNUEReader(io.BytesIO(bytes1), DEFAULT_FEATURES, config)
    model2 = reader1.model

    # 3. Export model2
    writer2 = NNUEWriter(model2, description=reader1.description, verbose=False)
    bytes2 = bytes(writer2.buf)

    assert bytes1 == bytes2, (
        f"Round-trip 1 mismatch: {len(bytes1)} vs {len(bytes2)} bytes"
    )

    # 4. Read back into model3
    reader2 = NNUEReader(io.BytesIO(bytes2), DEFAULT_FEATURES, config)
    model3 = reader2.model

    # 5. Export model3
    writer3 = NNUEWriter(model3, description=reader2.description, verbose=False)
    bytes3 = bytes(writer3.buf)

    assert bytes2 == bytes3, (
        f"Round-trip 2 mismatch: {len(bytes2)} vs {len(bytes3)} bytes"
    )

    # Verify model parameters match between model2 and model3
    for (n2, p2), (n3, p3) in zip(model2.named_parameters(), model3.named_parameters()):
        assert torch.equal(p2, p3), f"Parameter {n2} mismatch between model2 and model3"


def test_narrow_stacks_roundtrip():
    """--ls-widths: narrow stacks are written without their masked l1 outputs, read back with
    zeros there, and the model evaluates the same before and after."""
    torch.manual_seed(42)

    config = ModelConfig(L2=64, ls_widths="20:32,24:16")
    model = NNUEModel(feature_name=DEFAULT_FEATURES, config=config)
    init_random_weights(model)
    assert model.layer_stacks.l1_widths == [64] * 20 + [32] * 4 + [16] * 8

    bytes1 = bytes(NNUEWriter(model, description="Narrow stacks", verbose=False).buf)
    model2 = NNUEReader(io.BytesIO(bytes1), DEFAULT_FEATURES, config).model
    bytes2 = bytes(NNUEWriter(model2, description="Narrow stacks", verbose=False).buf)
    assert bytes1 == bytes2

    # Smaller than a uniform L2=64 net by the dropped rows and columns
    uniform = NNUEModel(feature_name=DEFAULT_FEATURES, config=ModelConfig(L2=64))
    init_random_weights(uniform)
    uniform_bytes = bytes(
        NNUEWriter(uniform, description="Narrow stacks", verbose=False).buf
    )
    assert len(bytes1) < len(uniform_bytes)

    # The masked l1 outputs and every weight reading them come back as zero
    ls = model2.layer_stacks
    w1 = ls.l1.linear.weight.view(ls.count, ls.L2, -1)
    assert w1[24, 14:62].abs().sum() == 0 and w1[24, 62:].abs().sum() > 0
    assert w1[20, 30:62].abs().sum() == 0 and w1[20, :30].abs().sum() > 0
    w2 = ls.l2.linear.weight.view(ls.count, ls.L3, -1)
    assert (
        w2[24, :, 14:62].abs().sum() == 0
        and w2[24, :, 64 + 14 : 64 + 62].abs().sum() == 0
    )

    # The masked outputs do not affect the evaluation: fill their weights with noise
    x = torch.rand(256, model.L1)
    idx = torch.arange(256) % ls.count
    with torch.no_grad():
        a = ls(x, idx, True, True)
        w1[24, 14:62] = torch.rand_like(w1[24, 14:62])
        w1[20, 30:62] = torch.rand_like(w1[20, 30:62])
        ls.l1.linear.bias.view(ls.count, ls.L2)[24, 14:62] = 0.5
        b = ls(x, idx, True, True)
    assert torch.equal(a, b)

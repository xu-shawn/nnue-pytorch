"""trainer.grad_clip against torch.nn.utils.clip_grad_norm_."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.grad_clip import clip_grad_norm_


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("max_norm", [1e-3, 2.0, 1e9])
def test_clip_matches_torch(device, max_norm):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3)
    shapes = [(1500, 1024), (7,), (64, 33), (1 << 20,)]
    ours = [torch.nn.Parameter(torch.zeros(s, device=device)) for s in shapes]
    theirs = [torch.nn.Parameter(torch.zeros(s, device=device)) for s in shapes]
    unused = torch.nn.Parameter(torch.zeros(3, device=device))
    for a, b in zip(ours, theirs):
        a.grad = torch.randn(a.shape, device=device) * 1e-3
        b.grad = a.grad.clone()
    norm = clip_grad_norm_([*ours, unused], max_norm)
    expected = torch.nn.utils.clip_grad_norm_([*theirs, unused], max_norm)
    assert torch.equal(norm, expected)
    for a, b in zip(ours, theirs):
        assert torch.equal(a.grad, b.grad)

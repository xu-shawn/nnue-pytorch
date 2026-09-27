"""Exact feature unions under collisions, overlap, and full compaction capacity."""

import pytest
import torch

from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required",
)
@pytest.mark.parametrize("batch", [1, 8, 17])
@pytest.mark.parametrize("active", [1, 33, 288])
@pytest.mark.parametrize("kind", ["empty", "overlap", "disjoint"])
def test_packed_feature_union(batch, active, kind):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 specialization")
    import cupy as cp
    import numpy as np

    from model.modules.feature_transformer.aggregated_ft_kernel import _kernels

    tiles = (batch + 7) // 8
    capacity = 16 * active
    hash_size = 1 << (capacity - 1).bit_length()
    rows = torch.full((batch, 2, active), -1, dtype=torch.int32)
    for position in range(batch):
        for perspective in range(2):
            if kind == "disjoint":
                # Distinct IDs with colliding hash starts; fill every slot.
                ids = torch.arange(active) + (2 * position + perspective) * active
                rows[position, perspective] = (ids % 7) + (ids // 7) * hash_size
            elif kind == "overlap":
                n = (position * 17 + active // 2) % (active + 1)
                ids = torch.arange(n) + position % 3 + perspective
                rows[position, perspective, :n] = (ids % 7) + (ids // 7) * hash_size

    white, black = (rows[:, i].contiguous().cuda() for i in range(2))
    ids = torch.empty((tiles, capacity), device="cuda", dtype=torch.int32)
    masks = torch.empty_like(ids)
    counts = torch.empty(tiles, device="cuda", dtype=torch.int32)
    pack, _, actual_capacity, shared_bytes, _ = _kernels(active, 1024)
    assert actual_capacity == capacity
    pack(
        (tiles,), (512,),
        (white.data_ptr(), black.data_ptr(), ids.data_ptr(), masks.data_ptr(),
         counts.data_ptr(), np.int32(batch)),
        shared_mem=shared_bytes,
        stream=cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream),
    )
    ids, masks, counts = ids.cpu(), masks.cpu(), counts.cpu()
    for tile in range(tiles):
        expected = {}
        for position in range(8 * tile, min(batch, 8 * (tile + 1))):
            for perspective in range(2):
                bit = 1 << (2 * (position % 8) + perspective)
                for feature in rows[position, perspective].tolist():
                    if feature >= 0:
                        expected[feature] = expected.get(feature, 0) | bit
        count = int(counts[tile])
        assert count == len(expected)
        actual = dict(zip(ids[tile, :count].tolist(), masks[tile, :count].tolist()))
        assert actual == expected

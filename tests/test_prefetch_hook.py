"""The loader's prefetch hook: FixedNumBatchesDataset runs it on device batches, NNUE builds it from
the feature transformer, and the feature set (not a constant) gives the tiled backward's list order."""

from types import SimpleNamespace

import pytest
import torch

from data_loader.dataset import FixedNumBatchesDataset
from model import NNUE
from model.modules import ComposedFeatureTransformer, get_feature_cls
from model.modules.feature_transformer import tiled_ft_prep
from model.quantize import QuantizationConfig, QuantizationManager


def test_list_buckets_from_feature_set():
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"),
        64,
        QuantizationManager(QuantizationConfig()),
    )
    # K32H3 is last: its rows start after Full_Threats + PP_3Wide, 768 rows per bucket.
    assert ft.list_buckets() == (64368, 768)


def test_list_buckets_without_bucketed_tail():
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats"), 64, QuantizationManager(QuantizationConfig())
    )
    assert ft.list_buckets() is None


def test_nnue_prefetch_hook_reorders_and_prepares():
    seen = []
    order = torch.tensor([2, 0, 1])
    fake = SimpleNamespace(
        model=SimpleNamespace(
            input=SimpleNamespace(
                prefetch_preparer=lambda: lambda w, b: seen.append((w, b)),
                locality_order=lambda w, b: order,
            )
        )
    )
    batch = tuple(torch.arange(3.0) + 10 * i for i in range(7))
    out = NNUE.prefetch_hook(fake)(batch)
    # every tensor of the batch is reordered alike, and the preparation is for the reordered one
    assert all(torch.equal(o, t[order]) for o, t in zip(out, batch))
    assert len(seen) == 1 and seen[0][0] is out[2] and seen[0][1] is out[3]

    # nothing to reorder by and nothing to prepare: the batch passes through
    fake.model.input.prefetch_preparer = lambda: None
    fake.model.input.locality_order = lambda w, b: None
    assert NNUE.prefetch_hook(fake)(batch) is batch


def test_locality_order_groups_positions_by_bucket():
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"),
        64,
        QuantizationManager(QuantizationConfig()),
    )
    first_row, rows_per_bucket = ft.list_buckets()
    gen = torch.Generator().manual_seed(0)
    white_bucket = torch.randint(0, 96, (200,), generator=gen)
    black_bucket = torch.randint(0, 96, (200,), generator=gen)

    def lists(bucket):
        # a few rows of other features, the list's bucket rows, then padding
        rows = torch.full((200, 8), -1, dtype=torch.int32)
        rows[:, 0] = torch.randint(0, first_row, (200,), generator=gen)
        rows[:, 1] = first_row + bucket * rows_per_bucket + 5
        rows[:, 2] = first_row + bucket * rows_per_bucket + 700
        return rows

    order = ft.locality_order(lists(white_bucket), lists(black_bucket))
    assert torch.equal(torch.sort(order)[0], torch.arange(200))
    key = white_bucket[order] * 96 + black_bucket[order]
    assert torch.all(key[1:] >= key[:-1])

    no_buckets = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats"), 64, QuantizationManager(QuantizationConfig())
    )
    assert no_buckets.locality_order(lists(white_bucket), lists(black_bucket)) is None


class _Batches:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield (torch.full((4,), float(i)), torch.zeros(2, 3, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hook_runs_on_device_batches_and_survives_errors():
    calls = []

    def hook(batch):
        calls.append(all(t.is_cuda for t in batch))
        if len(calls) == 2:
            raise RuntimeError("boom")
        return (batch[0] + 100.0, batch[1])

    ds = FixedNumBatchesDataset(
        _Batches(4), num_batches=4, device="cuda", prefetch_hook=hook
    )
    with pytest.warns(RuntimeWarning, match="prefetch hook failed"):
        out = [ds[i] for i in range(4)]
    # the first batch is the one the hook returned; the hook saw device tensors and was dropped
    # after its first error, and from then on batches arrive as they were
    assert [b[0][0].item() for b in out] == [100.0, 1.0, 2.0, 3.0]
    assert calls == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_parked_preparations_follow_their_batches(monkeypatch):
    """Every queued batch finds its preparation, however many are queued, and a batch that is
    freed unclaimed takes its entry with it."""

    def fake_prepare(white_indices, black_indices, buckets=None):
        t = torch.zeros(1, dtype=torch.int32, device=white_indices.device)
        return tiled_ft_prep.TiledPrep(2 * white_indices.shape[0], 1, t, t, t)

    monkeypatch.setattr(tiled_ft_prep, "prepare", fake_prepare)
    monkeypatch.setattr(tiled_ft_prep, "_parked", {})

    def batch():
        return torch.zeros(4, 8, dtype=torch.int32, device="cuda")

    queued = [batch() for _ in range(40)]
    for white in queued:
        tiled_ft_prep.register_prep(white, white)
    # claimed oldest first, as the trainer does, and only once
    assert all(tiled_ft_prep.claim_prep(white) is not None for white in queued)
    assert tiled_ft_prep.claim_prep(queued[0]) is None

    dropped = batch()
    tiled_ft_prep.register_prep(dropped, dropped)
    del dropped
    kept = batch()
    tiled_ft_prep.register_prep(kept, kept)
    assert len(tiled_ft_prep._parked) == 1
    assert tiled_ft_prep.claim_prep(kept) is not None

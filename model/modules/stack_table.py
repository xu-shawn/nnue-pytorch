"""Layer-stack fallback tables: which raw stacks (piece-count band x queen bucket x --ls-keys) get stored stacks,
and which stored stack every other raw stack uses.

A table is a list `remap` with remap[raw] = the raw stack whose weights `raw` uses; stacks with remap[s] == s are
stored. In configs and checkpoints it is written as "raw:fallback" pairs for the remapped stacks only.
"""

from pathlib import Path

import numpy as np
import torch

# Distance between key values when choosing a fallback (half: no pawns -> one half -> both halves; imb3:
# imbalanced -> balanced first)
KEY_DISTANCE = {
    "imb3": [[0, 1, 1], [1, 0, 2], [1, 2, 0]],
    "half": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    "offi": [[0, 1], [1, 0]],
}


def table_to_str(remap: list[int]) -> str:
    return ",".join(f"{s}:{f}" for s, f in enumerate(remap) if s != f)


def table_from_str(text: str, raw_count: int) -> list[int]:
    remap = list(range(raw_count))
    for pair in filter(None, text.split(",")):
        s, f = (int(v) for v in pair.split(":"))
        remap[s] = f
    return remap


def fallback_table(counts: np.ndarray, keys: list[str], sizes: dict[str, int], min_count: int) -> list[int]:
    """Route stacks seen fewer than min_count times to the nearest well-seen stack: same piece band and queen
    bucket with the fewest and smallest key changes, else the nearest piece band (ties: the more frequent)."""
    radix = int(np.prod([sizes[k] for k in keys])) if keys else 1
    count = len(counts)

    def split(s):
        b, r = divmod(s, radix)
        v = []
        for k in reversed(keys):
            r, x = divmod(r, sizes[k])
            v.append(x)
        return b, v[::-1]

    good = [s for s in range(count) if counts[s] >= min_count]
    assert good, "no layer stack reaches the minimum count"
    remap = list(range(count))
    for s in range(count):
        if counts[s] >= min_count:
            continue
        b, v = split(s)

        def score(t):
            bt, vt = split(t)
            band = abs(bt // 4 - b // 4) + (0 if bt % 4 == b % 4 else 100)
            changed = sum(x != y for x, y in zip(v, vt))
            dist = sum(KEY_DISTANCE[k][x][y] for k, x, y in zip(keys, v, vt))
            return band, changed, dist, -int(counts[t])

        remap[s] = min(good, key=score)
    return remap


def table_from_state_dict(state_dict) -> str | None:
    """The remap table saved in a checkpoint of a --ls-compact model, or None."""
    for k, v in state_dict.items():
        if k.endswith("layer_stacks.stack_fallback"):
            return table_to_str(v.tolist())
    return None


def use_checkpoint_stack_table(nnue_config, checkpoint: dict) -> None:
    """Before building a --ls-compact model to load a checkpoint into: take the checkpoint's layer-stack table."""
    mc = nnue_config.model_config
    if mc.ls_compact and not mc.ls_remap_table:
        table = table_from_state_dict(checkpoint["state_dict"])
        if table is not None:
            mc.ls_remap_table = table


def resolve_data_path(path: str, datasets: list[str]) -> str:
    """A binpack path as given, or relative to the data root of the training binpacks (data/<owner>/<repo>/<file>)."""
    p = Path(path)
    if p.exists():
        return str(p)
    for d in datasets:
        for root in Path(d).resolve().parents:
            if (root / path).exists():
                return str(root / path)
    raise FileNotFoundError(f"--ls-remap-data {path} not found (also tried relative to the training data)")


@torch.no_grad()
def count_stacks(binpack: str, features: str, model_config, threads: int, max_positions: int = 0) -> np.ndarray:
    """Raw layer-stack counts over a binpack (no position filtering), through the data loader and the model's own
    routing code. Only the K32 piece-square component is loaded."""
    from dataclasses import replace

    from data_loader.config import DataloaderSkipConfig
    from data_loader.stream import (
        create_sparse_batch_stream,
        destroy_sparse_batch,
        destroy_sparse_batch_stream,
        fetch_next_sparse_batch,
    )

    from ..model import NNUEModel

    k32 = [f for f in features.split("+") if f.startswith("K32")]
    assert len(k32) == 1, f"--ls-remap-data needs one K32 component in {features}"
    router = NNUEModel(
        k32[0], replace(model_config, L1=64, ls_compact=False, ls_remap="", ls_remap_table="", ls_factors="")
    )
    skip = DataloaderSkipConfig(
        filtered=False, wld_filtered=False, random_fen_skipping=0, early_fen_skipping=-1,
        soft_early_fen_skipping=0, pc_y0=1.0, pc_y1=1.0, pc_y2=1.0, pc_y3=1.0, pc_y4=1.0,
    )
    stream = create_sparse_batch_stream(
        router.input_feature_name.encode(), max(1, threads), [binpack], 65536, False, skip
    )
    counts = np.zeros(router.num_ls_buckets, np.int64)
    while not max_positions or counts.sum() < max_positions:
        b = fetch_next_sparse_batch(stream)
        if not b:
            break
        us, _, wi, _, _, _, pc, qb = b.contents.get_tensors("cpu")
        raw = router.calculate_buckets(pc, qb).view(-1) * router.ls_key_radix + router.stack_key_index(us, wi).view(-1)
        counts += np.bincount(raw.numpy(), minlength=router.num_ls_buckets)
        destroy_sparse_batch(b)
    destroy_sparse_batch_stream(stream)
    return counts

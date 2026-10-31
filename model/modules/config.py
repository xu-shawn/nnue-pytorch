from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
    ls_factors: Annotated[str, tyro.conf.arg(name="ls-factors")] = ""
    """Factorized layer stacks: ':'-separated factors, each a --ls-keys key or keys joined by '+' (e.g. "band+queen"),
    shared by the stacks with the same values of those keys, added to every stack layer during training and folded
    into the stacks at export (the exported net is unchanged in format). Empty = off."""
    ls_keys: Annotated[str, tyro.conf.arg(name="ls-keys")] = "band:queen"
    """Layer-stack keys, ':'-separated, in mixed radix with the last key varying fastest: "band" (piece-count band,
    (pieces - 1) / 4: 8), "queen" (2 if the side to move has a queen, plus 1 if the opponent has one: 4), "half" (pawns
    on both board halves a-d / e-h, on one half only, or none: 3), "offi" (1 if some side has two or more officers
    N, B, R, Q: 2). Stacks = the product of the key sizes. The data loader computes the stack index."""

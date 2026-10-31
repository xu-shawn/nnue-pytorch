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
    """Factorized layer stacks: ':'-separated factors shared by stacks with the same piece-count band ("band"), queen
    bucket ("queen"), band and queen bucket ("base") or --ls-keys value ("half", "offi"), added to every stack layer
    during training and folded into the stacks at export (the exported net is unchanged in format). Empty = off."""
    ls_keys: Annotated[str, tyro.conf.arg(name="ls-keys")] = ""
    """More layer-stack keys, ':'-separated, on top of piece-count band x queen bucket (mixed radix with the last key
    varying fastest): "half" (pawns on both board halves a-d / e-h, on one half only, or none: 3), "offi" (1 if some
    side has two or more officers N, B, R, Q: 2). Stacks = 32 x the product of the key sizes. The data loader computes
    the stack index."""

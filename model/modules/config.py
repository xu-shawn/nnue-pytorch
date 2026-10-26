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
    """Factorized layer stacks: ':'-separated factors shared by stacks with the same piece-count band ("band") or
    queen bucket ("queen"), added to every stack layer during training and folded into the stacks at export
    (the exported net is unchanged in format). Empty = off."""

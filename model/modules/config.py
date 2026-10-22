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
    ls_class_max: Annotated[int, tyro.conf.arg(name="ls-class-max")] = 0
    """Layer stacks for at most this many pieces (kings and pawns included; a multiple of 4,
    e.g. 12) are chosen by material class instead of the queen bucket: any queen, else any
    rook, else any minor piece, else pawns only. 0 = off."""

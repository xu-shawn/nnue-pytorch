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
    L2R: Annotated[int, tyro.conf.arg(name="l2-rank")] = 0
    """Factorised second hidden layer: when > 0, the sparse layer has L2R outputs and a
    per-stack dense L2R -> L2 mix (int16 weights) produces the L2 pre-activations. 0 = off."""

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
    material_rows: Annotated[bool, tyro.conf.arg(name="material-rows")] = False
    """Material rows in the feature transformer: one int16 row per piece type and side
    (relative to the perspective), added once per piece on the board, so material values
    need not fill the int8 piece-square weights. Needs a K32Q2-family feature."""

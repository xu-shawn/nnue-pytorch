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
    ls_widths: Annotated[str, tyro.conf.arg(name="ls-widths")] = ""
    """Narrower second hidden layer in some layer stacks: "<stack>:<width>[,...]". Stacks from
    <stack> on (the last entry that applies) have <width> l1 outputs, the last two being the skip
    outputs as usual; the other stacks have L2. Example with --l2=64: "20:32,24:16"."""

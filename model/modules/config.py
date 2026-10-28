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
    ls_imbalance: Annotated[bool, tyro.conf.arg(name="ls-imbalance")] = False
    """Also route the layer stacks by the side to move's material balance d (pawn units P1 N3 B3 R5 Q9) in 3
    classes: |d| <= 1, d >= 2, d <= -2. Gives 96 stacks, bucket = (piece-count band * 4 + queen bucket) * 3 +
    class. Needs a K32 piece-square component (the material is read from its planes)."""
    ls_keys: Annotated[str, tyro.conf.arg(name="ls-keys")] = ""
    """More layer-stack keys, ':'-separated, on top of piece-count band x queen bucket (after the --ls-imbalance
    class if that is set; mixed radix with the last key varying fastest): "imb3" (the --ls-imbalance class, 3),
    "half" (pawns on both board halves a-d / e-h, on one half only, or none: 3), "offi" (1 if some side has two or
    more officers N, B, R, Q: 2). Stacks = 32 x the product of the key sizes. Needs a K32 piece-square component."""
    ls_remap: Annotated[str, tyro.conf.arg(name="ls-remap")] = ""
    """File of "stack fallback" lines (a relative path that does not exist from the working directory resolves
    against the trainer directory). Listed stacks are routed to their fallback in training and exported as copies
    of it, so stacks without training data never reach the engine. The exported net format is unchanged. Empty =
    off."""
    ls_compact: Annotated[bool, tyro.conf.arg(name="ls-compact")] = False
    """With --ls-remap: store and export only the stacks that are their own fallback, in stack order. The net then
    has fewer layer stacks, and the engine maps raw stacks to stored ones with the same table."""

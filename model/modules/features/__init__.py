import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import tyro

from .full_threats import FullThreats, FullThreatsHm
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature
from .k32q2 import K32Q2
from .k32q2h2 import K32Q2H2
from .k32h3l import K32H3L
from .k32h3 import K32H3
from .k32q2p2 import K32Q2P2
from .k32q2p4 import K32Q2P4
from .pp_3wide import PP3Wide, PP3WideHm
from .ram_ram import RamRam, RamRamHm

_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "K32Q2^": K32Q2,
    "K32Q2H2^": K32Q2H2,
    "K32H3L^": K32H3L,
    "K32H3^": K32H3,
    "K32Q2P2^": K32Q2P2,
    "K32Q2P4^": K32Q2P4,
    "Full_Threats": FullThreats,
    "PP_3Wide": PP3Wide,
    "Ram_Ram": RamRam,
    # Oriented like HalfKAv2_hm (king on e..h); same sizes as the families above
    "Full_Threats_hm": FullThreatsHm,
    "PP_3Wide_hm": PP3WideHm,
    "Ram_Ram_hm": RamRamHm,
}

DEFAULT_FEATURES = "Full_Threats+PP_3Wide+K32Q2^"


def get_feature_cls(name: str) -> list[Callable[[int], InputFeature]]:
    parts = name.split("+")
    return [_FEATURE_COMPONENTS[p] for p in parts]


def get_available_features() -> list[str]:
    return list(_FEATURE_COMPONENTS.keys())


@dataclass(kw_only=True)
class FeatureConfig:
    features: Annotated[
        str,
        tyro.conf.arg(
            help="The feature set to use. Available: "
            + ", ".join(get_available_features())
            + ". Combine with +, e.g. Full_Threats+PP_3Wide+HalfKAv2_hm^"
        ),
    ] = DEFAULT_FEATURES


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default=DEFAULT_FEATURES,
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+PP_3Wide+HalfKAv2_hm^",
    )


__all__ = [
    "K32Q2",
    "FeatureConfig",
    "FullThreats",
    "HalfKav2Hm",
    "InputFeature",
    "PP3Wide",
    "add_feature_args",
    "get_available_features",
    "get_feature_cls",
]

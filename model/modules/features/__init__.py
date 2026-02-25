import argparse
from collections.abc import Callable

from .halfka_v2_hm import HalfKav2Hm
from .full_threats import FullThreats
from .input_feature import InputFeature
from .composed import ComposedFeatureTransformer, combine_input_features

_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threats": FullThreats,
}


def get_feature_cls(name: str) -> Callable[[int], ComposedFeatureTransformer]:
    parts = name.split("+")
    components = [_FEATURE_COMPONENTS[p] for p in parts]
    return combine_input_features(*components)


def get_available_features() -> list[str]:
    return list(_FEATURE_COMPONENTS.keys())


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="Full_Threats+HalfKAv2_hm^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+HalfKAv2_hm^",
    )


__all__ = [
    "HalfKav2Hm",
    "FullThreats",
    "InputFeature",
    "ComposedFeatureTransformer",
    "combine_input_features",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
]

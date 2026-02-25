import argparse

from .halfka_v2_hm import HalfKav2Hm
from .full_threats import FullThreats
from .input_feature import InputFeature
from .composed import ComposedFeatureTransformer, combine_input_features

_FEATURE_COMPONENTS: dict[str, type] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threats": FullThreats,
}

_FEATURES: dict[str, type] = {
    "HalfKAv2_hm^": combine_input_features(HalfKav2Hm),
    "Full_Threats": combine_input_features(FullThreats),
}


def get_feature_cls(name: str) -> type:
    if name in _FEATURES:
        return _FEATURES[name]

    if "+" in name:
        parts = name.split("+")
        components = [_FEATURE_COMPONENTS[p] for p in parts]
        return combine_input_features(*components)

    raise KeyError(f"Unknown feature '{name}'. Available: {', '.join(_FEATURES)}")


def get_available_features() -> list[str]:
    return list(_FEATURES.keys())


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

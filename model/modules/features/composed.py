import torch
from torch import nn

from ..feature_transformer import SparseLinearFunction
from .input_feature import InputFeature


class ComposedFeatureTransformer(nn.Module):
    """Thin coordinator that wraps one or more InputFeature modules.

    Each feature owns its own weight parameters. This class owns the shared
    bias and delegates everything else to the underlying features.
    """

    def __init__(self, features: list[InputFeature]):
        super().__init__()

        self.features = nn.ModuleList(features)
        self.num_outputs = features[0].num_outputs

        self.bias = nn.Parameter(torch.empty(self.num_outputs, dtype=torch.float32))

        # Aggregate attributes from components
        self.NUM_INPUTS = sum(f.NUM_INPUTS for f in features)
        self.MAX_ACTIVE_FEATURES = sum(f.MAX_ACTIVE_FEATURES for f in features)
        self.NUM_REAL_FEATURES = sum(f.NUM_REAL_FEATURES for f in features)

        self.FEATURE_NAME = "+".join(f.FEATURE_NAME for f in features)
        self.INPUT_FEATURE_NAME = "+".join(
            f.INPUT_FEATURE_NAME for f in features
        )

        self.HASH = self._compute_hash()

        self._reset_bias()

    def _compute_hash(self) -> int:
        h = 0
        for f in self.features:
            h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
            h ^= f.HASH
        return h

    def _reset_bias(self):
        import math

        sigma = math.sqrt(1 / self.NUM_INPUTS)
        with torch.no_grad():
            self.bias.uniform_(-sigma, sigma)

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        merged = torch.cat([f.merged_weight() for f in self.features], dim=0)
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                merged,
                self.bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                merged,
                self.bias,
            ),
        )

    @torch.no_grad()
    def coalesce(self) -> None:
        for f in self.features:
            f.coalesce()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        for f in self.features:
            f.init_weights(num_psqt_buckets, nnue2score)

        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.bias[L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return torch.cat([f.get_export_weights() for f in self.features], dim=0)

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        offset = 0
        for f in self.features:
            n = f.NUM_REAL_FEATURES
            f.load_export_weights(export_weight[offset : offset + n])
            offset += n

    def clip_weights(self, quantization) -> None:
        for f in self.features:
            f.clip_weights(quantization)


def combine_input_features(*feature_classes: type) -> type:
    """Return a factory class that creates a ComposedFeatureTransformer.

    The returned class has correct class-level attributes (HASH, NUM_INPUTS,
    FEATURE_NAME, etc.) so it can be used as a drop-in replacement for a
    single feature class in model.py.
    """
    num_inputs = sum(cls.NUM_INPUTS for cls in feature_classes)
    max_active = sum(cls.MAX_ACTIVE_FEATURES for cls in feature_classes)
    num_real = sum(cls.NUM_REAL_FEATURES for cls in feature_classes)

    feature_name = "+".join(cls.FEATURE_NAME for cls in feature_classes)
    input_feature_name = "+".join(
        cls.INPUT_FEATURE_NAME for cls in feature_classes
    )

    # Compute combined hash
    h = 0
    for cls in feature_classes:
        h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
        h ^= cls.HASH
    combined_hash = h

    class CombinedFeature:
        HASH = combined_hash
        FEATURE_NAME = feature_name
        INPUT_FEATURE_NAME = input_feature_name
        NUM_INPUTS = num_inputs
        MAX_ACTIVE_FEATURES = max_active
        NUM_REAL_FEATURES = num_real

        def __new__(cls, num_outputs: int) -> ComposedFeatureTransformer:
            features = [fc(num_outputs) for fc in feature_classes]
            return ComposedFeatureTransformer(features)

    CombinedFeature.__name__ = f"Combined_{'_'.join(c.__name__ for c in feature_classes)}"
    CombinedFeature.__qualname__ = CombinedFeature.__name__

    return CombinedFeature

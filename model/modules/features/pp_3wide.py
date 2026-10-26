import torch
from torch import nn

from .input_feature import InputFeature


class PP3Wide(InputFeature):
    HASH = 0x86F2B1DD
    FEATURE_NAME = "PP_3Wide"
    INPUT_FEATURE_NAME = "PP_3Wide"
    MAX_ACTIVE_FEATURES = 100

    NUM_INPUTS = 4560
    NUM_REAL_FEATURES = 4560
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)


class PP3WideHm(PP3Wide):
    """PP_3Wide oriented like HalfKAv2_hm (mirrored when the king is on files a..d, see
    FullThreatsHm). Same index space; the indices are a fixed permutation of PP_3Wide's."""

    HASH = 0x5C80FA69
    FEATURE_NAME = "PP_3Wide_hm"
    INPUT_FEATURE_NAME = "PP_3Wide_hm"

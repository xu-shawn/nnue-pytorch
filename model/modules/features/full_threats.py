import torch
from torch import nn

from .input_feature import InputFeature


class FullThreats(InputFeature):
    HASH = 0x2E6B9D04
    FEATURE_NAME = "Full_Threats"
    INPUT_FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128

    NUM_INPUTS = 59808
    NUM_REAL_FEATURES = 59808
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
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)


class FullThreatsHm(FullThreats):
    """Full_Threats oriented like HalfKAv2_hm: for each perspective the board is flipped vertically
    for black and mirrored horizontally when that perspective's king is on files a..d, so that the
    king lands on e..h (Full_Threats mirrors when the king is on e..h instead).

    Same index space; the indices are a fixed permutation of Full_Threats' (a horizontal mirror of
    both squares, with the attacker and the attacked piece swapped for the deduplicated same-type
    pairs on one rank). scripts/convert_hm_layout.py converts nets.
    """

    HASH = 0x2C41EAEA
    FEATURE_NAME = "Full_Threats_hm"
    INPUT_FEATURE_NAME = "Full_Threats_hm"

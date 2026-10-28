import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
from .modules.layer_stacks import STACK_KEY_STATES, stack_key_radix, stack_keys
from .quantize import QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.ls_keys = stack_keys(config)
        self.ls_key_radix = stack_key_radix(self.ls_keys)

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        # Index range of the K32 piece-square component, used to read the material for --ls-imbalance
        self.k32_range = None
        lo = 0
        for f in self.input.features:
            if type(f).__name__.startswith("K32"):
                self.k32_range = (lo, lo + f.NUM_INPUTS)
            lo += f.NUM_INPUTS
        assert not self.ls_keys or self.k32_range is not None, "--ls-imbalance / --ls-keys need a K32 piece-square component"
        self.layer_stacks = LayerStacks(32 * self.ls_key_radix, config, self.quantization)
        # Stacks stored in the net (fewer than the raw stacks with --ls-compact)
        self.num_ls_buckets = self.layer_stacks.count

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights()

    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input:
            self.input.clip_weights(self.quantization)

        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    if "stacked_layer" in group:
                        ge = group["stacked_layer"].group_expansion()
                        if ge is not None:
                            gw = ge[0].detach()
                            if not torch.is_tensor(min_weight):
                                min_weight = p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                            if not torch.is_tensor(max_weight):
                                max_weight = p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                            min_weight = min_weight - gw
                            max_weight = max_weight - gw
                    p_data_fp32.clamp_(min_weight, max_weight)

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.input.zero_virtual_weights()
        self.layer_stacks.zero_virtual_weights()

    def forward_ft(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> torch.Tensor:
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def calculate_buckets(self, piece_count: torch.Tensor, queen_bucket: torch.Tensor):
        pc_bucket = (piece_count - 1) // 4
        return pc_bucket * 4 + queen_bucket

    def imbalance_class(self, us: torch.Tensor, white_indices: torch.Tensor) -> torch.Tensor:
        """Side to move's material balance class: 0 |d| <= 1, 1 d >= 2, 2 d <= -2 (pawn units P1 N3 B3 R5 Q9).
        The counts come from the white perspective's K32 piece-square planes (768 rows per bucket, plane =
        piece type * 2 + (colour is black), square-major within a plane)."""
        lo, hi = self.k32_range
        inr = (white_indices >= lo) & (white_indices < hi)
        plane = torch.remainder(white_indices - lo, 768) // 64
        value = torch.tensor([1, -1, 3, -3, 3, -3, 5, -5, 9, -9, 0, 0], device=white_indices.device)
        d = torch.where(inr, value[plane.clamp(0, 11)], torch.zeros_like(plane)).sum(dim=1)
        d = torch.where(us.view(-1) > 0.5, d, -d)
        return torch.where(d.abs() <= 1, 0, torch.where(d > 0, 1, 2)).long()

    def _k32_planes(self, white_indices: torch.Tensor):
        lo, hi = self.k32_range
        inr = (white_indices >= lo) & (white_indices < hi)
        rel = torch.remainder(white_indices - lo, 768)
        return inr, rel // 64, rel % 64

    def half_class(self, white_indices: torch.Tensor) -> torch.Tensor:
        """0: pawns on both board halves (files a-d and e-h), 1: on one half only, 2: no pawns. The K32 horizontal
        mirror swaps the halves, which leaves the class unchanged."""
        inr, plane, sq = self._k32_planes(white_indices)
        pawn = inr & (plane < 2)
        q = (pawn & (sq % 8 <= 3)).any(dim=1)
        k = (pawn & (sq % 8 >= 4)).any(dim=1)
        return torch.where(q & k, 0, torch.where(q | k, 1, 2)).long()

    def officers_class(self, white_indices: torch.Tensor) -> torch.Tensor:
        """1 if some side has two or more officers (knights, bishops, rooks, queens), else 0."""
        inr, plane, _ = self._k32_planes(white_indices)
        officer = inr & (plane >= 2) & (plane < 10)
        white = (officer & (plane % 2 == 0)).sum(dim=1)
        black = (officer & (plane % 2 == 1)).sum(dim=1)
        return (torch.maximum(white, black) >= 2).long()

    def stack_key_index(self, us: torch.Tensor, white_indices: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros(white_indices.shape[0], dtype=torch.long, device=white_indices.device)
        for k in self.ls_keys:
            cls = {"imb3": lambda: self.imbalance_class(us, white_indices), "half": lambda: self.half_class(white_indices),
                   "offi": lambda: self.officers_class(white_indices)}[k]()
            idx = idx * STACK_KEY_STATES[k] + cls
        return idx

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        queen_bucket: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        layer_stack_indices = self.calculate_buckets(piece_count, queen_bucket)
        if self.ls_keys:
            layer_stack_indices = layer_stack_indices * self.ls_key_radix + self.stack_key_index(
                us, white_indices
            ).view_as(layer_stack_indices)
        layer_stack_indices = self.layer_stacks.stack_remap[layer_stack_indices]

        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        x = self.layer_stacks(
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights
        )

        return x

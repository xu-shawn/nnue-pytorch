import os

import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
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

        self.num_ls_buckets = 32

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        # Experiment: input features routed past the feature transformer into the outputs
        # of the first layer-stack layer, like a material input (NNUE_L2_FEATURES, loader
        # feature names joined by '+'). The loader appends their indices after the regular
        # ones; they need no accumulator, since 32-wide rows are cheap to sum per evaluation.
        l2_names = [n for n in os.environ.get("NNUE_L2_FEATURES", "").split("+") if n]
        self.l2_input = None
        if l2_names:
            from .modules.features import _FEATURE_COMPONENTS

            self.l2_offset = sum(f.NUM_INPUTS for f in self.input.features)
            l2_dims = sum(_FEATURE_COMPONENTS[n].NUM_INPUTS for n in l2_names)
            self.l2_input = nn.Linear(l2_dims, self.L2)
            with torch.no_grad():
                self.l2_input.weight.zero_()
                self.l2_input.bias.zero_()
            self.input_feature_name += "+" + "+".join(
                _FEATURE_COMPONENTS[n].INPUT_FEATURE_NAME for n in l2_names
            )

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

        extra = None
        if self.l2_input is not None:
            # Side-to-move view of the routed features, then drop them from the FT input
            # (they are the last valid entries of each row, so the rows stay packed)
            idx = torch.where(us.view(-1, 1) > 0.5, white_indices, black_indices).long()
            rel = idx - self.l2_offset
            rows, cols = torch.nonzero(rel >= 0, as_tuple=True)
            extra = self.l2_input.bias.expand(idx.shape[0], -1).clone()
            extra.index_add_(0, rows, self.l2_input.weight.t()[rel[rows, cols]])
            white_indices = torch.where(
                white_indices >= self.l2_offset, torch.full_like(white_indices, -1), white_indices
            )
            black_indices = torch.where(
                black_indices >= self.l2_offset, torch.full_like(black_indices, -1), black_indices
            )

        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        x = self.layer_stacks(
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights, extra
        )

        return x

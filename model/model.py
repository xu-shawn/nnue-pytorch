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

        # Material-class layer stacks (--ls-class-max): piece-type counts are read from the
        # piece-square indices (feature with a 768-input virtual factor: bucket * 768 +
        # plane * 64 + square, planes own/opponent P, N, B, R, Q, K)
        self.ls_class_max = config.ls_class_max
        assert self.ls_class_max % 4 == 0
        self.psq_offset, self.psq_span = 0, 0
        for f in self.input.features:
            if getattr(f, "NUM_INPUTS_VIRTUAL", 0) == 768:
                self.psq_span = f.NUM_INPUTS
                break
            self.psq_offset += f.NUM_INPUTS
        assert not self.ls_class_max or self.psq_span, "--ls-class-max needs K32Q2"

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

    def material_class(self, indices: torch.Tensor) -> torch.Tensor:
        """3 if any queen is on the board, else 2 if any rook, else 1 if any minor piece,
        else 0 (pawns only), from one perspective's piece-square indices."""
        rel = indices.long() - self.psq_offset
        is_psq = (indices >= 0) & (rel >= 0) & (rel < self.psq_span)
        plane = torch.where(is_psq, (rel % 768) // 64, torch.full_like(rel, 12))
        counts = torch.zeros(indices.shape[0], 13, device=indices.device)
        counts.scatter_add_(1, plane, torch.ones_like(plane, dtype=counts.dtype))
        q = counts[:, 8] + counts[:, 9]
        r = counts[:, 6] + counts[:, 7]
        m = counts[:, 2:6].sum(1)
        return torch.where(q > 0, 3, torch.where(r > 0, 2, torch.where(m > 0, 1, 0)))

    def calculate_buckets(
        self,
        piece_count: torch.Tensor,
        queen_bucket: torch.Tensor,
        indices: torch.Tensor | None = None,
    ):
        pc_bucket = (piece_count - 1) // 4
        if self.ls_class_max:
            assert indices is not None
            cls = (
                self.material_class(indices)
                .to(queen_bucket.dtype)
                .view_as(queen_bucket)
            )
            queen_bucket = torch.where(
                piece_count.view_as(queen_bucket) <= self.ls_class_max,
                cls,
                queen_bucket,
            )
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
        layer_stack_indices = self.calculate_buckets(
            piece_count, queen_bucket, white_indices if self.ls_class_max else None
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
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights
        )

        return x

import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
from .modules.feature_transformer.double_ft_functions import double_feature_transform
from .quantize import QuantizationManager

MATERIAL_ROWS_HASH = 0x4D41540A
# The fused FT kernels need unique indices within a row, so the k-th piece of a plane uses
# its own tied copy of the plane's material row.
MATERIAL_ROW_COPIES = 10


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

        # Material rows (--material-rows): one row per piece-square plane (own / opponent
        # P, N, B, R, Q, K), added once per piece. The planes are read from the K32Q2-family
        # indices (bucket * 768 + plane * 64 + square). The kings' rows are constant, so the
        # export folds them into the biases and the engine stores rows for P..Q only.
        self.psq_offset, self.psq_span = 0, 0
        for f in self.input.features:
            if getattr(f, "NUM_INPUTS_VIRTUAL", 0) == 768:
                self.psq_span = f.NUM_INPUTS
                break
            self.psq_offset += f.NUM_INPUTS
        self.mat_ft = None
        if config.material_rows:
            assert self.psq_span, "--material-rows needs a K32Q2-family feature"
            self.mat_ft = nn.Parameter(torch.zeros(12, self.L1))
            self.feature_hash ^= MATERIAL_ROWS_HASH

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
        if self.mat_ft is None:
            return self.input(
                us,
                them,
                white_indices,
                black_indices,
                fake_quantize_acts,
                fake_quantize_weights,
            )
        merged, bias = self.input.merged_weight_and_bias(fake_quantize_weights)
        n_in = merged.shape[0]
        mat = self.mat_ft.repeat_interleave(MATERIAL_ROW_COPIES, dim=0)
        if fake_quantize_weights:
            mat = self.quantization.fake_quantize_weights(mat, "ft_weight")
        merged = torch.cat([merged, mat], dim=0)

        def extend(idx):
            # Every piece-square index also activates its plane's material row. The fused
            # kernels read only the reserved width, and rows are packed (valid indices first,
            # then -1), so the extra indices go into the padding after the valid entries.
            rel = idx.long() - self.psq_offset
            is_psq = (idx >= 0) & (rel >= 0) & (rel < self.psq_span)
            plane = torch.where(is_psq, (rel % 768) // 64, torch.zeros_like(rel))
            oh = nn.functional.one_hot(plane, 12) * is_psq.unsqueeze(-1)
            copy = ((oh.cumsum(1) * oh).sum(-1) - 1).clamp(min=0)
            assert int(copy.max()) < MATERIAL_ROW_COPIES, "too many pieces of one type"
            val = n_in + plane * MATERIAL_ROW_COPIES + copy
            n_valid = (idx >= 0).sum(1, keepdim=True)
            pos = n_valid + is_psq.long().cumsum(1) - 1
            assert int((pos * is_psq).max()) < idx.shape[1], "no room for material rows"
            rows = (
                torch.arange(idx.shape[0], device=idx.device)
                .unsqueeze(1)
                .expand_as(pos)
            )
            out = idx.clone()
            out[rows[is_psq], pos[is_psq]] = val[is_psq].to(idx.dtype)
            return out

        l0_ = double_feature_transform(
            us,
            them,
            extend(white_indices),
            extend(black_indices),
            merged,
            bias,
            self.quantization.max_ft_activation,
            self.L1,
        )
        if fake_quantize_acts:
            l0_ = self.quantization.fake_quantize_ft_act(l0_)
        return l0_ * self.quantization.l0_correction_factor

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

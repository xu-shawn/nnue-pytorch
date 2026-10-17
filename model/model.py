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

        # Architecture experiments: options appended to NNUE_LS_VARIANT
        #   bkpc8 / bkpc16  layer stacks by piece count only (8 or 16 stacks, no queen split)
        #   psqt            a per-stack 768 -> 1 piece-square term added to the output
        #   tk              counts of the 80 threat kinds through 80 -> L3 into l2
        #   pn<P> / pnk<P>  pawn-structure MLP (side to move's oriented pawn planes, pnk also
        #                   both king squares) 128 -> P -> 32 added to the l1 outputs;
        #                   cacheable by pawn key in the engine. pn<P>x2: two hidden layers
        #   mat             material signature + count/difference one-hots -> 32 added to
        #                   the l1 outputs (changes only on captures and promotions)
        #   scale           drawishness: the output is multiplied by 2 * sigmoid(s), s linear
        #                   in the material features and pawn files (starts at exactly 1)
        _, *self.ls_options = os.environ.get("NNUE_LS_VARIANT", "base").split("+")
        self.bucket_mode = next(
            (o for o in self.ls_options if o.startswith("bk")), None
        )
        self.num_ls_buckets = {None: 32, "bkpc8": 8, "bkpc16": 16}[self.bucket_mode]

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        # Depth experiments: the att32p layer stacks also see the side to move's 768
        # piece-square inputs, taken from the K32Q2 indices modulo 768
        self.psq_offset = None
        offset = 0
        for f in self.input.features:
            if f.FEATURE_NAME.startswith("K32Q2"):
                self.psq_offset = offset
                self.psq_span = f.NUM_INPUTS
            offset += f.NUM_INPUTS

        # Threat kind of every threat feature index (80 = not a threat feature)
        if "tk" in self.ls_options:
            self.register_buffer("threat_kinds", _threat_kind_table(), persistent=False)
        else:
            self.threat_kinds = None

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
        if self.bucket_mode == "bkpc8":
            return (piece_count - 1) // 4
        if self.bucket_mode == "bkpc16":
            return (piece_count - 1) // 2
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
        psq = None
        tk = None
        if self.threat_kinds is not None:
            idx = torch.where(us.view(-1, 1) > 0.5, white_indices, black_indices).long()
            kinds = self.threat_kinds[idx.clamp(0, self.threat_kinds.shape[0] - 1)]
            is_threat = (idx >= 0) & (idx < self.threat_kinds.shape[0])
            kinds = torch.where(is_threat, kinds, torch.full_like(kinds, 80))
            tk = torch.zeros(idx.shape[0], 81, device=idx.device, dtype=l0_.dtype)
            tk.scatter_add_(1, kinds, torch.ones_like(kinds, dtype=l0_.dtype))
            tk = tk[:, :80]
        if (
            self.layer_stacks.variant
            in (
                "att32p",
                "att32pu",
            )
            or self.layer_stacks.block
            in (
                "g",
                "add",
            )
            or "psqt" in self.ls_options
            or "mat" in self.ls_options
            or "scale" in self.ls_options
            or any(o.startswith("pn") for o in self.ls_options)
        ):
            idx = torch.where(us.view(-1, 1) > 0.5, white_indices, black_indices).long()
            rel = idx - self.psq_offset
            valid = (idx >= 0) & (rel >= 0) & (rel < self.psq_span)
            cols = torch.where(valid, rel % 768, torch.full_like(rel, 768))
            psq = torch.zeros(idx.shape[0], 769, device=idx.device, dtype=l0_.dtype)
            psq.scatter_(1, cols, 1.0)
            psq = psq[:, :768]
        x = self.layer_stacks(
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights, psq, tk
        )

        return x


def _threat_kind_table() -> torch.Tensor:
    """Threat kind (attacker piece x target colour x target type, 80 kinds) of every
    Full_Threats index, following the loader's FullThreats layout."""
    targets = {"P": "NR", "N": "PNBRQ", "B": "PNBR", "R": "PNBR", "Q": "PNBRQ"}

    def n_attacks(pt, white, sq):
        f, r = sq % 8, sq // 8
        if pt == "P":
            if not 1 <= r <= 6:
                return 0
            return sum(1 for df in (-1, 1) if 0 <= f + df < 8)
        if pt == "N":
            jumps = (
                (1, 2),
                (2, 1),
                (-1, 2),
                (-2, 1),
                (1, -2),
                (2, -1),
                (-1, -2),
                (-2, -1),
            )
            return sum(1 for df, dr in jumps if 0 <= f + df < 8 and 0 <= r + dr < 8)
        diag, line = (
            [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            [(1, 0), (-1, 0), (0, 1), (0, -1)],
        )
        n = 0
        for df, dr in {"B": diag, "R": line, "Q": diag + line}[pt]:
            ff, rr = f + df, r + dr
            while 0 <= ff < 8 and 0 <= rr < 8:
                n += 1
                ff, rr = ff + df, rr + dr
        return n

    table = []
    kind = 0
    for white in (True, False):
        for pt in "PNBRQ":
            per_target = sum(n_attacks(pt, white, s) for s in range(64))
            for _ in range(2 * len(targets[pt])):
                table += [kind] * per_target
                kind += 1
    assert len(table) == 59808 and kind == 80, (len(table), kind)
    return torch.tensor(table, dtype=torch.long)

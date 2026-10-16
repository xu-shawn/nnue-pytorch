import os
from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class LayerStacks(nn.Module):
    def __init__(
        self, count: int, config: LayerStacksConfig, quantization: QuantizationManager
    ):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization

        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1 = FactorizedStackedLinear(
            2 * self.L1 // 2, self.L2, count, quantization, "ls_l1"
        )
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count, quantization, "ls_l2")

        # Depth experiments (proxy runs only; the serializer knows only "base"):
        #   base    the current layers, output sees both hidden layers plus a skip path
        #   noskip  output sees only the second hidden layer, no skip path (nettest #511)
        #   deepN   noskip plus N more 64 -> 32 hidden layers in a chain
        #   res64   64-wide stream: s = clip(s + act(W s)), act = sqr|crelu concat (64),
        #           3 blocks with their own weights (l2 is the first)
        #   rec64   as res64 but one block (l2) applied 3 times
        #   att32   32-wide stream: s = clip(s + crelu(W act(s)) * crelu(V x)), where V
        #           derives 32 fresh neurons from the feature transformer output x, 2 blocks
        #   att32p  as att32, but the fresh neurons come from the side to move's 768
        #           piece-square inputs through a 768 -> 32 layer shared by all stacks
        #           (an extra accumulator in the engine)
        self.variant = os.environ.get("NNUE_LS_VARIANT", "base")
        v = self.variant
        assert v in ("base", "noskip", "res64", "rec64", "att32", "att32p") or (
            v.startswith("deep") and v[4:].isdigit()
        ), v
        if v.startswith("deep"):
            extra = int(v[4:])
        elif v == "res64":
            extra = 2
        elif v in ("att32", "att32p"):
            extra = 1
        else:
            extra = 0
        self.extra = nn.ModuleList(
            StackedLinear(self.L3 * 2, self.L3, count, quantization, "ls_l2")
            for _ in range(extra)
        )
        # Fresh neurons from the feature transformer output, one set per att32 block
        self.fresh = nn.ModuleList(
            StackedLinear(self.L1, self.L3, count, quantization, "ls_l1")
            for _ in range(2 if v == "att32" else 0)
        )
        self.psq_fresh = nn.ModuleList(
            nn.Linear(768, self.L3) for _ in range(2 if v == "att32p" else 0)
        )

        self.output = StackedLinear(
            self.L2 * 2 + self.L3 * 2 if v == "base" else self.L3 * 2,
            1,
            count,
            quantization,
            "ls_output",
        )

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def _act(self, x: torch.Tensor, fake_quantize_acts: bool) -> torch.Tensor:
        """sqr-clipped-relu | clipped-relu concat, as used after l1 and l2."""
        sqr = torch.pow(x, 2.0)
        if fake_quantize_acts:
            sqr = self.quantization.fake_quantize_ls_act(sqr)
        sqr = sqr * (self.quantization.sqr_crelu_correction_factor)
        if fake_quantize_acts:
            x = self.quantization.fake_quantize_ls_act(x)
        return self.quantization.clip_ls_act(torch.cat([sqr, x], dim=1))

    def _crelu(self, x: torch.Tensor, fake_quantize_acts: bool) -> torch.Tensor:
        if fake_quantize_acts:
            x = self.quantization.fake_quantize_ls_act(x)
        return self.quantization.clip_ls_act(x)

    def _forward_variant(
        self, x, ls_indices, fake_quantize_acts, fake_quantize_weights, psq=None
    ):
        fq_a, fq_w = fake_quantize_acts, fake_quantize_weights
        l1c_ = self.l1(x, ls_indices, fq_w)
        v = self.variant

        if v == "noskip" or v.startswith("deep"):
            h = self._act(self.l2(self._act(l1c_, fq_a), ls_indices, fq_w), fq_a)
            for layer in self.extra:
                h = self._act(layer(h, ls_indices, fq_w), fq_a)
        elif v in ("res64", "rec64"):
            h = self._act(l1c_, fq_a)
            blocks = [self.l2, *self.extra] if v == "res64" else [self.l2] * 3
            for layer in blocks:
                h = self.quantization.clip_ls_act(
                    h + self._act(layer(h, ls_indices, fq_w), fq_a)
                )
        else:  # att32
            s = self._crelu(l1c_, fq_a)
            if v == "att32":
                fresh_out = [layer(x, ls_indices, fq_w) for layer in self.fresh]
            else:
                fresh_out = [layer(psq) for layer in self.psq_fresh]
            for layer, f_out in zip([self.l2, *self.extra], fresh_out):
                g = self._crelu(f_out, fq_a)
                u = self._crelu(layer(self._act(s, fq_a), ls_indices, fq_w), fq_a)
                prod = u * g
                if fq_a:
                    prod = self.quantization.fake_quantize_ls_act(prod)
                s = self.quantization.clip_ls_act(s + prod)
            h = self._act(s, fq_a)

        out = self.output(h, ls_indices, fq_w)
        if fq_a:
            out = self.quantization.fake_quantize_output(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
        psq: torch.Tensor | None = None,
    ):
        if self.variant != "base":
            return self._forward_variant(
                x, ls_indices, fake_quantize_acts, fake_quantize_weights, psq
            )

        l1c_ = self.l1(x, ls_indices, fake_quantize_weights)

        # Extract the short-path skip connection before fake quantization
        l1x_out = l1c_[:, -2].view(-1, 1) - l1c_[:, -1].view(-1, 1)
        l1x_ = l1c_

        l1_sqr = torch.pow(l1x_, 2.0)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)

        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)

        if fake_quantize_acts:
            l1x_ = self.quantization.fake_quantize_ls_act(l1x_)

        l1x_ = torch.cat([l1_sqr, l1x_], dim=1)
        l1x_ = self.quantization.clip_ls_act(l1x_)

        l2c_ = self.l2(l1x_, ls_indices, fake_quantize_weights)
        l2x_ = l2c_

        l2_sqr = torch.pow(l2x_, 2.0)
        if fake_quantize_acts:
            l2_sqr = self.quantization.fake_quantize_ls_act(l2_sqr)

        l2_sqr = l2_sqr * (self.quantization.sqr_crelu_correction_factor)

        if fake_quantize_acts:
            l2x_ = self.quantization.fake_quantize_ls_act(l2x_)

        l2x_ = torch.cat([l2_sqr, l2x_], dim=1)
        l2x_ = self.quantization.clip_ls_act(l2x_)

        l3_input = torch.cat([l1x_, l2x_], dim=1)

        l3c_ = self.output(l3_input, ls_indices, fake_quantize_weights)

        if fake_quantize_acts:
            l1x_out = self.quantization.fake_quantize_skip_act(l1x_out)

        l3x_ = l3c_ + l1x_out
        if fake_quantize_acts:
            l3x_ = self.quantization.fake_quantize_output(l3x_)

        assert l3x_.shape[1] == 1, (
            f"Expected output shape (batch_size, 1), got {l3x_.shape}"
        )
        return l3x_

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.l1.zero_virtual_weights()

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()

from collections.abc import Generator

import torch
import torch.nn.functional as F
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
        # Factorised l1 (--l2-rank > 0): the sparse layer has L2R outputs and a per-stack dense
        # L2R -> L2 mix produces the L2 pre-activations (the skip uses the last two mixed
        # outputs). The mix starts as a tiled identity plus noise, so each of the L2R
        # directions begins as several units with different offsets.
        self.L2R = config.L2R
        self.l1 = FactorizedStackedLinear(
            2 * self.L1 // 2, self.L2R or self.L2, count, quantization, "ls_l1"
        )
        self.l1mix = None
        if self.L2R:
            assert 0 < self.L2R <= 32 and self.L2R % 16 == 0 and self.L2 % 16 == 0
            self.l1mix = StackedLinear(self.L2R, self.L2, count, quantization, "ls_mix")
            with torch.no_grad():
                w = torch.empty(self.L2, self.L2R).uniform_(-0.05, 0.05)
                w[torch.arange(self.L2), torch.arange(self.L2) % self.L2R] += 1.0
                b = torch.empty(self.L2).uniform_(-0.1, 0.1)
                self.l1mix.linear.weight.copy_(w.repeat(count, 1))
                self.l1mix.linear.bias.copy_(b.repeat(count))
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count, quantization, "ls_l2")

        self.output = StackedLinear(
            self.L2 * 2 + self.L3 * 2, 1, count, quantization, "ls_output"
        )

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        l1c_ = self.l1(x, ls_indices, fake_quantize_weights)
        if self.l1mix is not None:
            # The engine saturates the mix input (int16 after >> 1)
            m = self.quantization.max_mix_input
            a = l1c_.clamp(-m, m)
            l1c_ = self.l1mix(a, ls_indices, fake_quantize_weights)
            if fake_quantize_acts:
                l1c_ = self._mix_engine_rounding(a, l1c_, ls_indices)

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

    def _mix_engine_rounding(
        self, a: torch.Tensor, y: torch.Tensor, ls_indices: torch.Tensor
    ) -> torch.Tensor:
        # The engine computes the mix in integers in the l1 output scale s (1.0 = s):
        # x16 = (x >> 1), y = (sum(x16 * w) >> 8) + bias, with w = round(512 * weight).
        # Reproduce that exactly in the forward value; the gradient goes through y.
        assert self.l1mix is not None
        q = self.quantization.weight_scales_dict
        s = q["ls_mix_bias"]
        with torch.no_grad():
            a16 = torch.floor(a * (s / 2) + 1e-3)
            w = torch.round(self.l1mix.linear.weight * q["ls_mix_weight"])
            b = torch.round(self.l1mix.linear.bias * s)
            acc = self.l1mix.select_output(F.linear(a16, w, b * 256), ls_indices)
            hard = torch.floor(acc / 256) / s
        return hard + (y - y.detach())

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
    def get_coalesced_mix(self) -> Generator[nn.Linear, None, None]:
        assert self.l1mix is not None
        for i in range(self.count):
            yield self.l1mix.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()

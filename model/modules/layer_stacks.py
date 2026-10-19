from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


def stack_l1_widths(count: int, L2: int, spec: str) -> list[int]:
    """Number of l1 outputs of each layer stack, from LayerStacksConfig.ls_widths."""
    widths = [L2] * count
    entries = [part.split(":") for part in spec.split(",") if part.strip()]
    for first, width in sorted((int(k), int(w)) for k, w in entries):
        if not 0 <= first < count or not 2 <= width <= L2:
            raise ValueError(f"bad ls-widths entry {first}:{width} (L2 = {L2})")
        widths[first:] = [width] * (count - first)
    return widths


def l1_kept_outputs(L2: int, width: int) -> list[int]:
    """The l1 outputs that a stack with `width` outputs keeps: the first width - 2 and the two
    skip outputs at the end."""
    return list(range(width - 2)) + [L2 - 2, L2 - 1]


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

        self.output = StackedLinear(
            self.L2 * 2 + self.L3 * 2, 1, count, quantization, "ls_output"
        )

        with torch.no_grad():
            self.output.linear.bias.zero_()

        # Narrower stacks (--ls-widths): the l1 outputs they do not have are zeroed here, and
        # the serializer leaves them out.
        self.l1_widths = stack_l1_widths(count, self.L2, config.ls_widths)
        self.narrow_stacks = any(w != self.L2 for w in self.l1_widths)
        l1_mask = torch.zeros(count, self.L2)
        for i, width in enumerate(self.l1_widths):
            l1_mask[i, l1_kept_outputs(self.L2, width)] = 1.0
        self.register_buffer("l1_mask", l1_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        l1c_ = self.l1(x, ls_indices, fake_quantize_weights)
        if self.narrow_stacks:
            l1c_ = l1c_ * self.l1_mask[ls_indices.view(-1).long()].to(l1c_.dtype)

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

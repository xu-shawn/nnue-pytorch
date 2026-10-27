from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


# Sizes of the extra layer-stack keys (see LayerStacksConfig.ls_keys)
STACK_KEY_STATES = {"imb3": 3, "half": 3, "offi": 2}


def stack_keys(config) -> list[str]:
    """The extra layer-stack keys in index order: --ls-imbalance's class first, then --ls-keys."""
    keys = (["imb3"] if getattr(config, "ls_imbalance", False) else []) + [
        k for k in getattr(config, "ls_keys", "").split(":") if k
    ]
    for k in keys:
        assert k in STACK_KEY_STATES, f"unknown layer-stack key {k!r} (known: {sorted(STACK_KEY_STATES)})"
    assert len(set(keys)) == len(keys), f"repeated layer-stack key in {keys}"
    return keys


def stack_key_radix(keys: list[str]) -> int:
    radix = 1
    for k in keys:
        radix *= STACK_KEY_STATES[k]
    return radix


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

        factors = [f for f in getattr(config, "ls_factors", "").split(":") if f]
        if factors:
            self.set_group_factors(self.factor_maps(count, factors, stack_keys(config)))

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
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

    @staticmethod
    def factor_maps(count: int, names: list[str], keys: list[str] | None = None) -> list[torch.Tensor]:
        """Group id of every stack under each named factor: "band", "queen", "base" (the 32-stack bucket) and each
        extra key by name ("imb" is an alias of "imb3"). bucket = (piece-count band * 4 + queen bucket) * radix +
        the mixed-radix index of the extra keys, the last varying fastest."""
        keys = keys or []
        per = stack_key_radix(keys)
        assert count == 32 * per, (count, keys)
        s = torch.arange(count)
        base = s // per
        table = {"band": base // 4, "queen": base % 4, "base": base}
        rest = s % per
        for k in reversed(keys):
            table[k] = rest % STACK_KEY_STATES[k]
            rest = rest // STACK_KEY_STATES[k]
        if "imb3" in table:
            table["imb"] = table["imb3"]
        for n in names:
            assert n in table, f"unknown layer-stack factor {n!r} (known: {sorted(table)})"
        return [table[n] for n in names]

    def set_group_factors(self, maps: list[torch.Tensor]) -> None:
        for layer in (self.l1, self.l2, self.output):
            layer.set_group_factors(maps)

    def group_factor_params(self, biases: bool) -> list:
        out = []
        for layer in (self.l1, self.l2, self.output):
            out += list(layer.group_factor_b if biases else layer.group_factor_w)
        return out

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
        self.l2.coalesce_group_factors()
        self.output.coalesce_group_factors()

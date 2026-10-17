import os
import re
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
        # NNUE_LS_VARIANT is "<variant>[+option...]"; options: fact (as NNUE_LS_FACT=1),
        # l1s<x> (as NNUE_L1_SCALE=x)
        variant, *options = os.environ.get("NNUE_LS_VARIANT", "base").split("+")
        # Block kind of the hidden layers after l1: c (64 -> 64 crelu), s (64 -> 64
        # sqr-crelu), g (psq gate), glu (self-gated), add (psq added); None for the
        # plain sqr|crelu layers. Used by the resN<x> and deepN<x> variants.
        m = re.fullmatch(r"s?deep\d+(c|s|g|glu|add1|add)i?", variant)
        # deepNadd1: one 768 -> L3 piece-square layer shared by all hidden layers (one
        # 32-wide accumulator in the engine instead of one per layer)
        self.psq_shared = m is not None and m.group(1) == "add1"
        self.block = (
            m.group(1).rstrip("1")
            if m
            else {
                "resc3": "c",
                "ress3": "s",
                "resg3": "g",
                "resglu3": "glu",
                "resadd3": "add",
            }.get(variant)
        )
        block_out = {"c": 2 * self.L3, "s": 2 * self.L3, "glu": 4 * self.L3}.get(
            self.block, self.L3
        )
        # NNUE_LS_FACT=1: l2 and the output layer also get a component shared by all
        # stacks, like l1 (coalesced at export, so free at inference). Most of the 32
        # stacks see under 1% of the positions each.
        self.ls_fact = os.environ.get("NNUE_LS_FACT", "0") == "1" or "fact" in options
        # NNUE_L1_SCALE: scale of the l1 outputs entering the activations (the skip path
        # is unchanged); 0.5 is one more right shift in the engine, halving saturation
        self.l1_scale = float(os.environ.get("NNUE_L1_SCALE", "1.0"))
        for o in options:
            if o.startswith("l1s"):
                self.l1_scale = float(o[3:])
            else:
                assert o in ("fact", "psqt", "tk", "bkpc8", "bkpc16"), o
        # +psqt: per-stack linear piece-square term on the output (an int32 accumulator
        # with one lane per stack in the engine); +tk: threat-kind counts into l2
        self.psqt = nn.Linear(768, count) if "psqt" in options else None
        self.tk_layer = nn.Linear(80, self.L3) if "tk" in options else None
        with torch.no_grad():
            for layer in (self.psqt, self.tk_layer):
                if layer is not None:
                    layer.weight.zero_()
                    layer.bias.zero_()
        ls_cls = FactorizedStackedLinear if self.ls_fact else StackedLinear
        self.l2 = ls_cls(self.L2 * 2, block_out, count, quantization, "ls_l2")

        # Depth experiments (proxy runs only; the serializer knows only "base"):
        #   base    the current layers, output sees both hidden layers plus a skip path
        #   noskip  output sees only the second hidden layer, no skip path (nettest #511)
        #   deepN   noskip plus N more 64 -> 32 hidden layers in a chain
        #   deepN..i  with the hidden layers after l1 starting at bias 0.25, so that their
        #           activations are not all rounded to zero at initialisation
        #   deepNadd1  deepNadd with one piece-square layer shared by all hidden layers
        #   sdeepN..  as deepN.. but dense like the base net: the output sees the l1
        #           activations and every hidden layer's, plus the skip path
        #   deepNx  deepN with every hidden layer after l1 of block kind x (c, s, g, glu,
        #           add; see the res variants below for what each block does)
        #   res64   64-wide stream: s = clip(s + act(W s)), act = sqr|crelu concat (64),
        #           3 blocks with their own weights (l2 is the first)
        #   resN    as res64 with N blocks
        #   rec64   as res64 but one block (l2) applied 3 times
        #   att32   32-wide stream: s = clip(s + crelu(W act(s)) * crelu(V x)), where V
        #           derives 32 fresh neurons from the feature transformer output x, 2 blocks
        #   att32p  as att32, but the fresh neurons come from the side to move's 768
        #           piece-square inputs through a 768 -> 32 layer shared by all stacks
        #           (an extra accumulator in the engine)
        #   att32pu as att32p, but each layer stack has its own 768 -> 32 layer (int8,
        #           computed from the ~32 active inputs at evaluation in the engine)
        #   resc3   res64 without the 32 bottleneck: 3 blocks s = clip(s + crelu(W s)), W 64 -> 64
        #   ress3   as resc3 with sqr-clipped-relu
        #   resg3   res64 with a gate from the 768 piece-square inputs:
        #           s = clip(s + act(W s) * crelu(V psq)), V 768 -> 64 shared by all stacks
        #   resglu3 self-gated blocks: s = clip(s + crelu(A s) * crelu(B s)), A, B 64 -> 64
        #   resadd3 piece-square neurons added instead of multiplied:
        #           s = clip(s + act(W s + V psq)), V 768 -> 32 shared by all stacks
        self.variant = variant
        v = self.variant
        assert (
            v
            in (
                "base",
                "noskip",
                "res64",
                "rec64",
                "att32",
                "att32p",
                "att32pu",
                "resc3",
                "ress3",
                "resg3",
                "resglu3",
                "resadd3",
            )
            or re.fullmatch(r"s?deep\d+(c|s|g|glu|add1|add)?i?", v) is not None
            or (v.startswith("res") and v[3:].isdigit())
            or re.fullmatch(r"res\d+d", v) is not None
        ), v
        self.dense = v.startswith("sdeep")
        if v.startswith(("deep", "sdeep")):
            extra = int(re.match(r"s?deep(\d+)", v).group(1))
        elif v == "res64":
            extra = 2
        elif re.fullmatch(r"res\d+d", v):
            extra = int(v[3:-1]) - 1
        elif v.startswith("res") and v[3:].isdigit():
            extra = int(v[3:]) - 1
        elif v in ("resc3", "ress3", "resg3", "resglu3", "resadd3"):
            extra = 2
        elif v in ("att32", "att32p", "att32pu"):
            extra = 1
        else:
            extra = 0
        self.extra = nn.ModuleList(
            StackedLinear(self.L3 * 2, block_out, count, quantization, "ls_l2")
            for _ in range(extra)
        )
        # Fresh neurons from the feature transformer output, one set per att32 block
        self.fresh = nn.ModuleList(
            StackedLinear(
                self.L1 if v == "att32" else 768, self.L3, count, quantization, "ls_l1"
            )
            for _ in range(2 if v in ("att32", "att32pu") else 0)
        )
        n_psq, w_psq = (2, self.L3) if v == "att32p" else (0, 0)
        if self.block == "g":
            n_psq, w_psq = 1 + extra, 2 * self.L3
        elif self.block == "add":
            n_psq, w_psq = 1 if self.psq_shared else 1 + extra, self.L3
        self.psq_fresh = nn.ModuleList(nn.Linear(768, w_psq) for _ in range(n_psq))
        # resNd: a signed 64 -> 64 projection after each block's activation, so the
        # residual update can be negative
        self.proj = nn.ModuleList(
            StackedLinear(self.L3 * 2, self.L3 * 2, count, quantization, "ls_l2")
            for _ in range(1 + extra if re.fullmatch(r"res\d+d", v) else 0)
        )
        if re.fullmatch(r"s?deep\d+(c|s|g|glu|add1|add)?i", v):
            with torch.no_grad():
                for layer in [self.l2, *self.extra]:
                    layer.linear.bias.fill_(0.25)
        # Saturation statistics of the residual stream, recorded during validation
        self.stream_stats = {}
        if self.block == "g":
            # Start with the gates open (crelu(1) = 127/128), so the gated layers begin
            # as plain ones and learn where to close
            with torch.no_grad():
                for gate in self.psq_fresh:
                    gate.bias.fill_(1.0)

        out_in = self.L3 * 2
        if v == "base":
            out_in = self.L2 * 2 + self.L3 * 2
        elif self.dense:
            out_in = self.L2 * 2 + (1 + extra) * self.L3 * 2
        self.output = ls_cls(
            out_in,
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

    def _block(self, layer, k, h, ls_indices, psq, fq_a, fq_w) -> torch.Tensor:
        """Hidden layer k of kind self.block applied to the 64-wide activations h."""
        b = self.block
        if b == "add":
            v = self.psq_fresh[0 if self.psq_shared else k](psq)
            return self._act(layer(h, ls_indices, fq_w) + v, fq_a)
        t = layer(h, ls_indices, fq_w)
        if b == "c":
            return self._crelu(t, fq_a)
        if b == "s":
            return self._sqr(t, fq_a)
        if b == "glu":
            # Products of two values in [0, 1) mostly fall below one activation step
            # (1/128) and the chain dies; scale by 4 (a 2-bit smaller shift in int8)
            half = t.shape[1] // 2
            a = self._crelu(t[:, :half], fq_a) * self._crelu(t[:, half:], fq_a)
            a = self.quantization.clip_ls_act(4 * a)
        else:  # g
            a = self._act(t, fq_a) * self._crelu(self.psq_fresh[k](psq), fq_a)
        if fq_a:
            a = self.quantization.fake_quantize_ls_act(a)
        return a

    @torch.compiler.disable
    @torch.no_grad()
    def _record_stream(self, k, h, a, new) -> None:
        """Per block: fraction of stream units at the clip, and of the update lost to it.

        Kept as device tensors (no host sync) and outside torch.compile, so that
        validation is not slowed down or recompiled.
        """
        top = self.quantization.max_hidden_activation
        sat = (new >= top - 1e-6).float().mean()
        lost = (h + a - new).abs().sum() / a.abs().sum().clamp_min(1e-9)
        st = self.stream_stats.get(k)
        if st is None:
            self.stream_stats[k] = [sat, lost, 1]
        else:
            st[0] = st[0] + sat
            st[1] = st[1] + lost
            st[2] += 1

    def print_stream_stats(self) -> None:
        for k, (sat, lost, n) in sorted(self.stream_stats.items()):
            print(
                f"STREAM block {k}: {100 * sat.item() / n:.1f}% of units at the clip, "
                f"{100 * lost.item() / n:.1f}% of the update lost to clipping ({n} batches)",
                flush=True,
            )
        self.stream_stats.clear()

    def _sqr(self, x: torch.Tensor, fake_quantize_acts: bool) -> torch.Tensor:
        """sqr-clipped-relu alone, scaled like the first half of _act."""
        sqr = torch.pow(x, 2.0)
        if fake_quantize_acts:
            sqr = self.quantization.fake_quantize_ls_act(sqr)
        return self.quantization.clip_ls_act(
            sqr * self.quantization.sqr_crelu_correction_factor
        )

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

        if self.dense:
            h = self._act(l1c_ * self.l1_scale, fq_a)
            hs = [h]
            for k, layer in enumerate([self.l2, *self.extra]):
                if self.block is None:
                    h = self._act(layer(h, ls_indices, fq_w), fq_a)
                else:
                    h = self._block(layer, k, h, ls_indices, psq, fq_a, fq_w)
                hs.append(h)
            skip = l1c_[:, -2:-1] - l1c_[:, -1:]
            out = self.output(torch.cat(hs, dim=1), ls_indices, fq_w)
            if fq_a:
                skip = self.quantization.fake_quantize_skip_act(skip)
            out = out + skip
            if fq_a:
                out = self.quantization.fake_quantize_output(out)
            return out

        if v.startswith("deep") and self.block is not None:
            h = self._act(l1c_, fq_a)
            for k, layer in enumerate([self.l2, *self.extra]):
                h = self._block(layer, k, h, ls_indices, psq, fq_a, fq_w)
        elif v == "noskip" or v.startswith("deep"):
            h = self._act(self.l2(self._act(l1c_, fq_a), ls_indices, fq_w), fq_a)
            for layer in self.extra:
                h = self._act(layer(h, ls_indices, fq_w), fq_a)
        elif v in ("resc3", "ress3", "resglu3", "resg3", "resadd3"):
            h = self._act(l1c_, fq_a)
            blocks = [self.l2, *self.extra]
            for k, layer in enumerate(blocks):
                h = self.quantization.clip_ls_act(
                    h + self._block(layer, k, h, ls_indices, psq, fq_a, fq_w)
                )
        elif v == "rec64" or v.startswith("res"):
            h = self._act(l1c_, fq_a)
            blocks = [self.l2] * 3 if v == "rec64" else [self.l2, *self.extra]
            for k, layer in enumerate(blocks):
                a = self._act(layer(h, ls_indices, fq_w), fq_a)
                if self.proj:
                    a = self.proj[k](a, ls_indices, fq_w)
                    if fq_a:
                        a = self.quantization.fake_quantize_ls_act(a)
                new = self.quantization.clip_ls_act(h + a)
                if not self.training:
                    self._record_stream(k, h, a, new)
                h = new
        else:  # att32
            s = self._crelu(l1c_, fq_a)
            if v == "att32":
                fresh_out = [layer(x, ls_indices, fq_w) for layer in self.fresh]
            elif v == "att32pu":
                fresh_out = [layer(psq, ls_indices, fq_w) for layer in self.fresh]
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
        tk: torch.Tensor | None = None,
    ):
        if self.variant != "base":
            assert self.psqt is None and self.tk_layer is None
            return self._forward_variant(
                x, ls_indices, fake_quantize_acts, fake_quantize_weights, psq
            )

        l1c_ = self.l1(x, ls_indices, fake_quantize_weights)

        # Extract the short-path skip connection before fake quantization
        l1x_out = l1c_[:, -2].view(-1, 1) - l1c_[:, -1].view(-1, 1)
        l1x_ = l1c_ * self.l1_scale if self.l1_scale != 1.0 else l1c_

        l1_sqr = torch.pow(l1x_, 2.0)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)

        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)

        if fake_quantize_acts:
            l1x_ = self.quantization.fake_quantize_ls_act(l1x_)

        l1x_ = torch.cat([l1_sqr, l1x_], dim=1)
        l1x_ = self.quantization.clip_ls_act(l1x_)

        l2c_ = self.l2(l1x_, ls_indices, fake_quantize_weights)
        if self.tk_layer is not None:
            l2c_ = l2c_ + self.tk_layer(tk)
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
        if self.psqt is not None:
            l3x_ = l3x_ + self.psqt(psq).gather(1, ls_indices.view(-1, 1).long())
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

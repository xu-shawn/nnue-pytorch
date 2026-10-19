import operator
import struct
from collections.abc import Sequence
from functools import reduce
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ..config import ModelConfig
from ..model import NNUEModel
from ..modules.layer_stacks import l1_kept_outputs


def ascii_hist(name, x, bins=7):
    start, end = int(x.min()), int(x.max())
    if start >= end - bins:
        start -= (bins + 1) // 2
        end += bins // 2
    edges = np.linspace(start, end + 1, bins + 1).astype(int)
    edges = np.unique(edges)
    N, X = np.histogram(x, bins=edges)
    width = 50
    nmax = N.max()

    print(name)
    for xi, n in zip(X, N):
        bar = "#" * int(n * 1.0 * width / nmax)
        xi = f"{xi: <8.4g}".ljust(10)
        print(f"{xi}| {bar}")


def get_histogram_callback(hist_title: str, verbose: bool):
    if not verbose:
        return None

    def histogram_callback(
        hist_subtitle: str,
        values: torch.Tensor,
    ):
        total_elements = values.numel()
        hist_desc = [hist_title, hist_subtitle]
        hist_desc = " ".join(filter(None, hist_desc))

        if total_elements == 0:
            print(f"Layer '{hist_desc}' is empty.")
            return

        print("-" * 15)

        min_value = values.min().item()
        num_argmin = int((values == min_value).sum().item())
        max_value = values.max().item()
        num_argmax = int((values == max_value).sum().item())

        ascii_hist(f"{hist_desc}: ", values.detach().cpu().numpy())
        print(f"Number of elements: {total_elements}")
        print(f"Minimum value in layer is {min_value}, occurring {num_argmin} times.")
        print(f"Maximum value in layer is {max_value}, occurring {num_argmax} times.")
        print("-" * 15)

    return histogram_callback


# hardcoded for now
VERSION = 0x6A448AFA
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."


def narrow_indices(
    L2: int, width: int, output_inputs: int
) -> tuple[list[int], list[int], list[int]]:
    """For a layer stack with `width` l1 outputs out of L2: the l1 rows it keeps, the l2 input
    columns (squared, then clipped activations) and the output layer's input columns."""
    rows = l1_kept_outputs(L2, width)
    cols = rows + [L2 + i for i in rows]
    out_cols = cols + list(range(2 * L2, output_inputs))
    return rows, cols, out_cols


@torch.no_grad()
def narrow_layer_stack(
    l1: nn.Linear, l2: nn.Linear, output: nn.Linear, L2: int, width: int
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """The layers of a stack with `width` l1 outputs, without the outputs it does not have."""
    rows, cols, out_cols = narrow_indices(L2, width, output.in_features)

    def linear(weight: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
        return layer

    return (
        linear(l1.weight[rows], l1.bias[rows]),
        linear(l2.weight[:, cols], l2.bias),
        linear(output.weight[:, out_cols], output.bias),
    )


class NNUEWriter:
    """
    All values are stored in little endian.
    """

    def __init__(
        self,
        model: NNUEModel,
        description: str | None = None,
        verbose: bool = True,
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION

        self.verbose = verbose

        fc_hash = self.fc_hash(model)

        # 1. Plaintext header
        self.buf = bytearray()
        self.write_header(model, fc_hash, description)

        # 2. Payload (Feature Transformer + Layer Stacks)
        self.int32(model.feature_hash ^ (model.L1 * 2))  # Feature transformer hash
        self.write_feature_transformer(model)
        layer_stacks = model.layer_stacks
        for bucket, (l1, l2, output) in enumerate(
            layer_stacks.get_coalesced_layer_stacks()
        ):
            width = layer_stacks.l1_widths[bucket]
            if width != model.L2:
                l1, l2, output = narrow_layer_stack(l1, l2, output, model.L2, width)
            self.int32(self.fc_hash(model, width))  # FC layers hash
            self.write_fc_layer(
                model, l1, layer_stacks.l1.layer_key, f"bucket {bucket}"
            )
            self.write_fc_layer(
                model, l2, layer_stacks.l2.layer_key, f"bucket {bucket}"
            )
            self.write_fc_layer(
                model, output, layer_stacks.output.layer_key, f"bucket {bucket}"
            )

    @staticmethod
    def fc_hash(model: NNUEModel, l1_width: int | None = None) -> int:
        """Hash of a layer stack; l1_width overrides the number of l1 outputs (L2)."""
        # InputSlice hash
        prev_hash = 0xEC42E90D
        prev_hash ^= model.L1 * 2

        # Fully connected layers
        layers = [
            model.layer_stacks.l1.linear,
            model.layer_stacks.l2.linear,
            model.layer_stacks.output.linear,
        ]
        out_features = [layer.out_features // model.num_ls_buckets for layer in layers]
        if l1_width is not None:
            out_features[0] = l1_width
        for num_outputs in out_features:
            layer_hash = 0xCC03DAE4
            layer_hash += num_outputs
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            if num_outputs != 1:
                # Clipped ReLU hash
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash
        return layer_hash

    def write_header(self, model: NNUEModel, fc_hash: int, description: str) -> None:
        self.int32(VERSION)  # version
        self.int32(fc_hash ^ model.feature_hash ^ (model.L1 * 2))  # halfkp network hash
        encoded_description = description.encode("utf-8")
        self.int32(len(encoded_description))  # Network definition
        self.buf.extend(encoded_description)

    def write_tensor(self, arr: torch.Tensor) -> None:
        arr = arr.detach().flatten().cpu().numpy()
        self.buf.extend(arr.tobytes())

    def write_feature_transformer(self, model: NNUEModel) -> None:
        layer = model.input

        bias = layer.bias.data[: model.L1]

        # Get export weights (coalesced + remapped 12→11 piece types)
        weight = layer.get_export_weights()

        # biases are exported as i16s
        biases = model.quantization.quantize_feature_transformer_bias(
            bias, get_histogram_callback("", self.verbose)
        )

        self.write_tensor(biases)

        # Weights stored as [num_features][outputs]
        offset = 0
        for f in layer.features:
            n = f.NUM_REAL_FEATURES
            f_export_dtype = f.EXPORT_WEIGHT_DTYPE

            ft_histogram_callback = get_histogram_callback(f.FEATURE_NAME, self.verbose)
            segment_weight = weight[offset : offset + n]
            segment_weight = model.quantization.quantize_feature_transformer_weights(
                segment_weight, f_export_dtype, ft_histogram_callback
            )
            offset += n

            self.write_tensor(segment_weight)

    def write_fc_layer(
        self,
        model: NNUEModel,
        layer: nn.Linear,
        layer_key: str | None,
        desc: str,
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        bias = layer.bias.data
        weight = layer.weight.data

        if layer_key is None:
            raise RuntimeError("layer_key required for quantization.")

        bias, weight = model.quantization.quantize_fc_layer(
            bias, weight, layer_key, get_histogram_callback(desc, self.verbose)
        )

        # FC inputs are padded to 32 elements by spec.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, : weight.shape[1]] = weight
            weight = new_w

        self.write_tensor(bias)
        # Weights stored as [outputs][inputs], so we can flatten
        self.write_tensor(weight)

    def int32(self, v: int) -> None:
        self.buf.extend(struct.pack("<I", v))


class NNUEReader:
    def __init__(
        self,
        f: BinaryIO,
        feature_name: str,
        config: ModelConfig,
    ):
        self.f = f
        self.feature_name = feature_name
        self.model = NNUEModel(feature_name, config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        self.read_header(self.model.feature_hash, fc_hash)
        self.read_int32(
            self.model.feature_hash ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.model.zero_virtual_weights()

        self.read_feature_transformer(self.model.input)

        layers = [
            self.model.layer_stacks.l1,
            self.model.layer_stacks.l2,
            self.model.layer_stacks.output,
        ]
        num_ls_buckets = self.model.num_ls_buckets
        l_w_slices = [
            torch.chunk(layer.linear.weight.data, num_ls_buckets, dim=0)
            for layer in layers
        ]
        l_b_slices = [
            torch.chunk(layer.linear.bias.data, num_ls_buckets, dim=0)
            for layer in layers
        ]

        L2 = self.model.L2
        for b in range(num_ls_buckets):
            width = self.model.layer_stacks.l1_widths[b]
            self.read_int32(NNUEWriter.fc_hash(self.model, width))  # FC layers hash
            if width == L2:
                for layer_idx in range(len(layers)):
                    self.read_fc_layer(
                        l_w_slices[layer_idx][b],
                        l_b_slices[layer_idx][b],
                        layers[layer_idx].layer_key,
                    )
                continue
            # A narrower stack: read its layers, then place them in the full-width layers.
            # The l1 outputs it does not have, and the weights reading them, stay zero.
            (l1_w, l2_w, out_w) = (l_w_slices[i][b] for i in range(3))
            (l1_b, l2_b, out_b) = (l_b_slices[i][b] for i in range(3))
            rows, cols, out_cols = narrow_indices(L2, width, out_w.shape[1])
            w1, b1 = torch.zeros(len(rows), l1_w.shape[1]), torch.zeros(len(rows))
            w2 = torch.zeros(l2_w.shape[0], len(cols))
            w3 = torch.zeros(1, len(out_cols))
            self.read_fc_layer(w1, b1, layers[0].layer_key)
            self.read_fc_layer(w2, l2_b, layers[1].layer_key)
            self.read_fc_layer(w3, out_b, layers[2].layer_key)
            for full in (l1_w, l1_b, l2_w, out_w):
                full.zero_()
            l1_w[rows], l1_b[rows] = w1, b1
            l2_w[:, cols] = w2
            out_w[:, out_cols] = w3

    def read_header(self, feature_hash: int, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        self.read_int32(fc_hash ^ feature_hash ^ (self.config.L1 * 2))
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def tensor(self, dtype: npt.DTypeLike, shape: Sequence[int]) -> torch.Tensor:
        count = reduce(operator.mul, shape, 1)
        itemsize = np.dtype(dtype).itemsize
        raw = self.f.read(count * itemsize)
        if len(raw) != count * itemsize:
            raise EOFError(
                f"Unexpected end of file: expected {count * itemsize} bytes, got {len(raw)}"
            )
        d = np.frombuffer(raw, dtype=dtype)
        d = torch.from_numpy(d.astype(np.float32))
        d = d.reshape(shape)
        return d

    def read_feature_transformer(self, layer) -> None:
        L1 = layer.num_outputs

        bias = self.tensor(np.int16, [L1])
        segments = []

        for feature in layer.features:
            dtype = np.int8 if feature.EXPORT_WEIGHT_DTYPE == torch.int8 else np.int16
            s = self.tensor(dtype, [feature.NUM_REAL_FEATURES, L1])
            segments.append(s)

        weight = torch.cat(segments, dim=0)

        bias, weight = self.model.quantization.dequantize_feature_transformer(
            bias, weight
        )

        layer.bias.data = bias.to(torch.float32)
        layer.load_export_weights(weight.to(torch.float32))

    def read_fc_layer(
        self,
        layer_weight_t: torch.Tensor,
        layer_bias_t: torch.Tensor,
        layer_key: str,
    ) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer_weight_t.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        bias = self.tensor(np.int32, layer_bias_t.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, layer_key
        )

        layer_bias = bias.to(torch.float32)
        # Strip padding.
        layer_weight = weight[: non_padded_shape[0], : non_padded_shape[1]].to(
            torch.float32
        )

        layer_bias_t.data.copy_(layer_bias)
        layer_weight_t.data.copy_(layer_weight)

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise ValueError(f"Expected: {expected:x}, got: {v:x}")
        return v

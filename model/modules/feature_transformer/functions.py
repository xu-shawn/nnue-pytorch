import torch
from torch import autograd

from .kernel import (
    sparse_input_linear_backward,
    sparse_input_linear_forward,
)


class SparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        # active_counts is computed as a byproduct of the forward kernel
        active_counts = torch.empty(
            batch_size, dtype=torch.int32, device=device
        )

        sparse_input_linear_forward(
            feature_indices,
            feature_values,
            weight,
            bias,
            output,
            active_counts,
            batch_size,
            max_active_features,
            output_size
        )

        ctx.save_for_backward(feature_indices, feature_values, weight, bias, active_counts)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias, active_counts = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.empty(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.empty(output_size, dtype=torch.float32, device=device)

        sparse_input_linear_backward(
            feature_indices,
            feature_values,
            weight_grad,
            bias_grad,
            grad_output,
            active_counts,
            batch_size,
            max_active_features,
            output_size
        )

        return None, None, weight_grad, bias_grad

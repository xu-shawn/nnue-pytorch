import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 32}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 1024}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 2048}),
    ],
    key=["max_active_indices", "output_size"]
)
@triton.jit
def sparse_input_linear_forward_kernel(
        input_indices,
        input_values,
        weight,
        bias,
        output,
        active_counts,
        max_active_indices: tl.constexpr,
        output_size: tl.constexpr,
        OUTPUT_BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    output_block_idx = tl.program_id(1)

    output_offsets = OUTPUT_BLOCK_SIZE * output_block_idx + tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = output_offsets < output_size

    input_indices_slice = input_indices + batch_idx * max_active_indices
    input_values_slice = input_values + batch_idx * max_active_indices
    output_slice = output + batch_idx * output_size

    acc = tl.load(bias + output_offsets, mask=output_mask, other=0.0)
    acc = acc.to(tl.float32)

    num_active = 0
    past_active_indices = False
    for k in range(max_active_indices):
        if not past_active_indices:
            feature_idx = tl.load(input_indices_slice + k)
            if feature_idx == -1:
                past_active_indices = True
            else:
                num_active += 1
                curr_feature_values = tl.load(input_values_slice + k)
                curr_weight_values = tl.load(weight + feature_idx * output_size + output_offsets, mask=output_mask, other=0.0)
                acc += curr_weight_values * curr_feature_values

    tl.store(output_slice + output_offsets, acc, mask=output_mask)

    # Store active count (only from block 0 to avoid redundant writes)
    if output_block_idx == 0:
        tl.store(active_counts + batch_idx, num_active)


def sparse_input_linear_forward(
        input_indices,
        input_values,
        weight,
        bias,
        output,
        active_counts,
        batch_size,
        max_active_indices,
        output_size
):
    def grid(meta):
        return (batch_size, triton.cdiv(output_size, meta["OUTPUT_BLOCK_SIZE"]))

    sparse_input_linear_forward_kernel[grid](
        input_indices=input_indices,
        input_values=input_values,
        weight=weight,
        bias=bias,
        output=output,
        active_counts=active_counts,
        max_active_indices=max_active_indices,
        output_size=output_size,
    )


def _zero_weight_grad(nargs):
    nargs["weight_grad"].zero_()


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 32}, num_warps=1, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}, num_warps=1, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}, num_warps=2, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}, num_warps=2, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}, num_warps=4, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}, num_warps=4, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}, num_warps=8, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}, num_warps=8, pre_hook=_zero_weight_grad),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}, num_warps=16, pre_hook=_zero_weight_grad),
    ],
    key=["max_active_indices", "output_size"],
)
@triton.jit
def sparse_input_linear_backward_kernel(
        input_indices,
        input_values,
        weight_grad,
        output_grad,
        active_counts,
        max_active_indices: tl.constexpr,
        output_size: tl.constexpr,
        OUTPUT_BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    output_block_idx = tl.program_id(1)

    output_offsets = OUTPUT_BLOCK_SIZE * output_block_idx + tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = output_offsets < output_size

    feature_indices_slice = input_indices + batch_idx * max_active_indices
    feature_values_slice = input_values + batch_idx * max_active_indices

    output_grad_slice = output_grad + batch_idx * output_size
    output_grad_values = tl.load(output_grad_slice + output_offsets, mask=output_mask, other=0.0)
    nonzero_grad_mask = output_mask & (output_grad_values != 0)

    num_active = tl.load(active_counts + batch_idx)
    k = 0
    while k < num_active:
        feature_idx = tl.load(feature_indices_slice + k)
        curr_feature_values = tl.load(feature_values_slice + k)
        curr_weight_grad_values = output_grad_values * curr_feature_values
        tl.atomic_add(
            weight_grad + feature_idx * output_size + output_offsets,
            curr_weight_grad_values,
            mask=nonzero_grad_mask
        )
        k += 1


def sparse_input_linear_backward(
        input_indices,
        input_values,
        weight_grad,
        bias_grad,
        output_grad,
        active_counts,
        batch_size,
        max_active_indices,
        output_size
):
    def grid(meta):
        return (
            batch_size,
            triton.cdiv(output_size, meta['OUTPUT_BLOCK_SIZE'])
        )

    sparse_input_linear_backward_kernel[grid](
        input_indices=input_indices,
        input_values=input_values,
        weight_grad=weight_grad,
        output_grad=output_grad,
        active_counts=active_counts,
        max_active_indices=max_active_indices,
        output_size=output_size
    )

    # bias_grad via efficient reduction instead of per-element atomics
    import torch
    torch.sum(output_grad, dim=0, out=bias_grad)

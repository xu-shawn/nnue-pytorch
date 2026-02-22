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

    past_active_indices = False
    for k in range(max_active_indices):
        if not past_active_indices:
            feature_idx = tl.load(input_indices_slice + k)
            if feature_idx == -1:
                past_active_indices = True
            else:
                curr_feature_values = tl.load(input_values_slice + k)
                curr_weight_values = tl.load(weight + feature_idx * output_size + output_offsets, mask=output_mask, other=0.0)
                acc += curr_weight_values * curr_feature_values

    tl.store(output_slice + output_offsets, acc, mask=output_mask)


def sparse_input_linear_forward(
        input_indices,
        input_values,
        weight,
        bias,
        output,
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
        max_active_indices=max_active_indices,
        output_size=output_size,
    )


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 8}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 16}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 32}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 1024}),
    ],
    key=["max_active_indices", "output_size"]
)
@triton.jit
def sparse_input_linear_backward_kernel(
        input_indices,
        input_values,
        bias_grad,
        weight_grad,
        output_grad,
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

    tl.atomic_add(
        bias_grad + output_offsets,
        output_grad_values,
        mask=nonzero_grad_mask
    )

    past_active_indices = False
    k = 0
    while k < max_active_indices and not past_active_indices:
        feature_idx = tl.load(feature_indices_slice + k)
        if feature_idx == -1:
            past_active_indices = True
        else:
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
        bias_grad,
        weight_grad,
        output_grad,
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
        bias_grad=bias_grad,
        output_grad=output_grad,
        max_active_indices=max_active_indices,
        output_size=output_size
    )

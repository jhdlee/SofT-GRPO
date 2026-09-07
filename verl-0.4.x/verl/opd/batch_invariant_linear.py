"""Opt-in, fixed-tile linear projection for the Qwen replay diagnostic.

The CUDA forward uses one fixed 32 x 64 x 32 Triton tile, FP32 accumulation,
and no split-K or autotuning. Row counts and projection dimensions are runtime
arguments, so prefill/decode and packed/separate projections use the same
reduction order. This is a candidate for batch invariance, not a claim of
validated model-level replay parity. CPU forward delegates to ``F.linear``;
CPU tests therefore establish interface/gradient behavior only.

CUDA supports matching BF16/FP16 tensors on one NVIDIA device and TP=1 only.
Backward uses tiled PyTorch matrix products with FP32 accumulation (FP64
for CPU double inputs). It never promotes the full token-by-vocabulary
upstream gradient or full weight matrix. Split reductions can change FP32
rounding compared with a single dense GEMM; forward arithmetic is unchanged.
CUDA backward requires TF32 to have been disabled by the caller. No parameters
are installed, replaced, or modified by this API.
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU diagnostics do not require a Triton installation.
    triton = None
    tl = None


# Bound all three matrix axes. In the Qwen LM head K=1024, so its largest
# upstream FP32 tile is 32 MiB and its weight/gradient accumulator is 16 MiB.
# Returned gradients keep the input dtype; GEMM library workspace is separate.
_BACKWARD_ROW_CHUNK = 2048
_BACKWARD_OUTPUT_CHUNK = 4096
_BACKWARD_INPUT_CHUNK = 4096


def _flat_view(tensor):
    """Flatten leading dimensions only when it is a view, never a dense copy."""
    try:
        return tensor.view(math.prod(tensor.shape[:-1]), tensor.shape[-1])
    except RuntimeError:
        return None


def _matrix_tile(tensor, flat_view, row_start, row_stop, column_start, column_stop):
    if flat_view is not None:
        return flat_view[row_start:row_stop, column_start:column_stop]
    # A permuted higher-dimensional upstream gradient may not flatten as a
    # view. Index the requested rows and columns together so that even this
    # fallback copies only one tile, not the whole token-by-vocabulary matrix.
    rows = torch.arange(row_start, row_stop, device=tensor.device)
    coordinates = []
    for dimension in reversed(tensor.shape[:-1]):
        coordinates.append(rows.remainder(dimension))
        rows = rows.div(dimension, rounding_mode="floor")
    return tensor[(*reversed(coordinates), slice(column_start, column_stop))]


if triton is not None:

    @triton.jit(
        do_not_specialize=["M", "N", "K"],
        do_not_specialize_on_alignment=["X", "W", "B", "Y", "M", "N", "K"],
    )
    def _linear_kernel(X, W, B, Y, M, N, K, HAS_BIAS: tl.constexpr):
        # Keep both the tile and launch settings fixed across every shape.
        # Widen before multiplication: packed Qwen vocabulary outputs can
        # exceed 2**31 elements although M, N and K each fit in int32. Pointer
        # addition cannot repair an already-wrapped row/column stride product.
        rows = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32).to(tl.int64)
        columns = tl.program_id(1).to(tl.int64) * 64 + tl.arange(0, 64).to(tl.int64)
        reduction = tl.arange(0, 32).to(tl.int64)
        accumulator = tl.zeros((32, 64), dtype=tl.float32)
        for block in range(tl.cdiv(K, 32)):
            indices = block.to(tl.int64) * 32 + reduction
            x = tl.load(
                X + rows[:, None] * K + indices[None, :],
                mask=(rows[:, None] < M) & (indices[None, :] < K),
                other=0.0,
            )
            # W is contiguous [N, K]; access it as the transposed RHS.
            weight = tl.load(
                W + columns[None, :] * K + indices[:, None],
                mask=(columns[None, :] < N) & (indices[:, None] < K),
                other=0.0,
            )
            accumulator = tl.dot(x, weight, accumulator, input_precision="ieee")
        if HAS_BIAS:
            bias = tl.load(B + columns, mask=columns < N, other=0.0)
            accumulator = accumulator + bias[None, :].to(tl.float32)
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            accumulator.to(Y.dtype.element_ty),
            mask=(rows[:, None] < M) & (columns[None, :] < N),
        )


def _validate(value, weight, bias, tensor_parallel_size):
    if type(tensor_parallel_size) is not int or tensor_parallel_size != 1:
        raise ValueError("batch_invariant_linear supports tensor_parallel_size=1 only")
    if value.ndim < 1 or weight.ndim != 2:
        raise ValueError("expected input with at least one dimension and a [N, K] weight")
    if value.shape[-1] != weight.shape[1]:
        raise ValueError("input and weight feature dimensions differ")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have shape [N]")
    tensors = (value, weight) if bias is None else (value, weight, bias)
    if any(t.device != value.device or t.dtype != value.dtype for t in tensors):
        raise ValueError("input, weight, and bias must have matching devices and dtypes")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("batch_invariant_linear requires real floating point tensors")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError("batch_invariant_linear supports CPU or NVIDIA CUDA tensors only")
    if value.is_cuda:
        if torch.version.hip is not None:
            raise ValueError("batch_invariant_linear CUDA diagnostic requires NVIDIA CUDA")
        if value.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("CUDA batch_invariant_linear requires BF16 or FP16 tensors")
        if triton is None:
            raise RuntimeError("CUDA batch_invariant_linear requires Triton")


class _BatchInvariantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        with torch.autocast(device_type=value.device.type, enabled=False):
            if not value.is_cuda:
                return F.linear(value, weight, bias)

            rows = math.prod(value.shape[:-1])
            features = weight.shape[0]
            output = torch.empty((*value.shape[:-1], features), device=value.device, dtype=value.dtype)
            if rows == 0 or features == 0:
                return output
            x = value.reshape(rows, value.shape[-1]).contiguous()
            w = weight.contiguous()
            b = bias.contiguous() if bias is not None else w
            # A device guard also supports a caller whose current device differs.
            with torch.cuda.device(value.device):
                _linear_kernel[(triton.cdiv(rows, 32), triton.cdiv(features, 64))](
                    x, w, b, output, rows, features, value.shape[-1],
                    HAS_BIAS=bias is not None, num_warps=4, num_stages=2,
                )
            return output

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        needs_x, needs_w, needs_b = ctx.needs_input_grad
        if value.is_cuda and (needs_x or needs_w) and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("CUDA batch_invariant_linear backward requires matmul.allow_tf32=False")
        accumulation_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        rows = math.prod(value.shape[:-1])
        outputs, inputs = weight.shape
        with torch.autocast(device_type=value.device.type, enabled=False):
            gradient_view = _flat_view(grad_output)
            value_view = _flat_view(value)
            grad_value = grad_weight = grad_bias = None
            if needs_x:
                grad_value_matrix = torch.empty((rows, inputs), device=value.device, dtype=value.dtype)
                for row_start in range(0, rows, _BACKWARD_ROW_CHUNK):
                    row_stop = min(row_start + _BACKWARD_ROW_CHUNK, rows)
                    for input_start in range(0, inputs, _BACKWARD_INPUT_CHUNK):
                        input_stop = min(input_start + _BACKWARD_INPUT_CHUNK, inputs)
                        accumulator = None
                        for output_start in range(0, outputs, _BACKWARD_OUTPUT_CHUNK):
                            output_stop = min(output_start + _BACKWARD_OUTPUT_CHUNK, outputs)
                            gradient = _matrix_tile(grad_output, gradient_view, row_start, row_stop, output_start, output_stop).to(accumulation_dtype)
                            weight_tile = weight[output_start:output_stop, input_start:input_stop].to(accumulation_dtype)
                            if accumulator is None:
                                accumulator = gradient @ weight_tile
                            else:
                                accumulator.addmm_(gradient, weight_tile)
                            del gradient, weight_tile
                        if accumulator is None:
                            grad_value_matrix[row_start:row_stop, input_start:input_stop].zero_()
                        else:
                            grad_value_matrix[row_start:row_stop, input_start:input_stop] = accumulator.to(value.dtype)
                            del accumulator
                grad_value = grad_value_matrix.reshape(value.shape)
            if needs_w:
                grad_weight = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
                for output_start in range(0, outputs, _BACKWARD_OUTPUT_CHUNK):
                    output_stop = min(output_start + _BACKWARD_OUTPUT_CHUNK, outputs)
                    for input_start in range(0, inputs, _BACKWARD_INPUT_CHUNK):
                        input_stop = min(input_start + _BACKWARD_INPUT_CHUNK, inputs)
                        accumulator = None
                        for row_start in range(0, rows, _BACKWARD_ROW_CHUNK):
                            row_stop = min(row_start + _BACKWARD_ROW_CHUNK, rows)
                            gradient = _matrix_tile(grad_output, gradient_view, row_start, row_stop, output_start, output_stop).to(accumulation_dtype)
                            value_tile = _matrix_tile(value, value_view, row_start, row_stop, input_start, input_stop).to(accumulation_dtype)
                            if accumulator is None:
                                accumulator = gradient.transpose(0, 1) @ value_tile
                            else:
                                accumulator.addmm_(gradient.transpose(0, 1), value_tile)
                            del gradient, value_tile
                        if accumulator is None:
                            grad_weight[output_start:output_stop, input_start:input_stop].zero_()
                        else:
                            grad_weight[output_start:output_stop, input_start:input_stop] = accumulator.to(weight.dtype)
                            del accumulator
            if needs_b and ctx.has_bias:
                grad_bias = torch.empty(outputs, device=value.device, dtype=value.dtype)
                for output_start in range(0, outputs, _BACKWARD_OUTPUT_CHUNK):
                    output_stop = min(output_start + _BACKWARD_OUTPUT_CHUNK, outputs)
                    accumulator = None
                    for row_start in range(0, rows, _BACKWARD_ROW_CHUNK):
                        row_stop = min(row_start + _BACKWARD_ROW_CHUNK, rows)
                        gradient = _matrix_tile(grad_output, gradient_view, row_start, row_stop, output_start, output_stop).to(accumulation_dtype)
                        contribution = gradient.sum(dim=0)
                        accumulator = contribution if accumulator is None else accumulator + contribution
                        del gradient, contribution
                    if accumulator is None:
                        grad_bias[output_start:output_stop].zero_()
                    else:
                        grad_bias[output_start:output_stop] = accumulator.to(value.dtype)
                        del accumulator
        return grad_value, grad_weight, grad_bias


def batch_invariant_linear(value, weight, bias=None, *, tensor_parallel_size=1):
    """Apply ``value @ weight.T + bias`` without replacing model parameters.

    CUDA forward requires matching BF16/FP16 tensors, including bias. Autocast
    does not change this contract: callers must explicitly provide that dtype.
    TP1 is an explicit diagnostic restriction, not distributed synchronization.
    """
    _validate(value, weight, bias, tensor_parallel_size)
    return _BatchInvariantLinear.apply(value, weight, bias)

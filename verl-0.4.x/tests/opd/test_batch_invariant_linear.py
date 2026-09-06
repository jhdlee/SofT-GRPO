"""CPU contract/gradient checks and conditional CUDA arithmetic comparisons."""

import pytest
import torch
import torch.nn.functional as F

from verl.opd.batch_invariant_linear import batch_invariant_linear


@pytest.mark.parametrize("with_bias", [False, True])
def test_double_gradcheck_input_weight_and_bias(with_bias):
    torch.manual_seed(901)
    value = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
    weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    bias = torch.randn(4, dtype=torch.double, requires_grad=True)
    arguments = (value, weight, bias) if with_bias else (value, weight)
    assert torch.autograd.gradcheck(batch_invariant_linear, arguments)


@pytest.mark.parametrize("shape,features", [((35, 37), 67), ((2, 3, 5, 37), 67), ((37,), 65)])
def test_cpu_arbitrary_leading_dimensions_and_noncontiguous_tails(shape, features):
    torch.manual_seed(19)
    value = torch.randn(*shape, 2, dtype=torch.double)[..., 0].requires_grad_()
    weight = torch.randn(shape[-1], features, dtype=torch.double).T.requires_grad_()
    bias = torch.randn(features * 2, dtype=torch.double)[::2].requires_grad_()
    assert not value.is_contiguous() and not weight.is_contiguous() and not bias.is_contiguous()
    actual = batch_invariant_linear(value, weight, bias)
    expected = F.linear(value, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (value, weight, bias), upstream)
    expected_grads = torch.autograd.grad(expected, (value, weight, bias), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape,features", [((0, 5), 7), ((2, 0, 5), 7), ((3, 0), 7), ((3, 5), 0), ((0,), 7)])
def test_empty_rows_features_and_reduction_have_correct_gradients(shape, features):
    value = torch.randn(shape, dtype=torch.double, requires_grad=True)
    weight = torch.randn(features, shape[-1], dtype=torch.double, requires_grad=True)
    bias = torch.randn(features, dtype=torch.double, requires_grad=True)
    output = batch_invariant_linear(value, weight, bias)
    expected = F.linear(value, weight, bias)
    assert output.shape == (*shape[:-1], features)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(output.sum(), (value, weight, bias))
    expected_grads = torch.autograd.grad(expected.sum(), (value, weight, bias))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_backward_accumulates_in_fp32_and_preserves_dtype_under_autocast(dtype):
    torch.manual_seed(12)
    value = torch.randn(2, 33, 35).to(dtype).requires_grad_()
    weight = torch.randn(67, 35).to(dtype).requires_grad_()
    bias = torch.randn(67).to(dtype).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = batch_invariant_linear(value, weight, bias)
        gradient = torch.randn_like(output)
        output.backward(gradient)
    x = value.detach().reshape(-1, 35).float()
    g = gradient.reshape(-1, 67).float()
    expected = ((g @ weight.detach().float()).reshape_as(value).to(dtype),
                (g.T @ x).to(dtype), g.sum(0).to(dtype))
    assert output.dtype == dtype
    for tensor, reference in zip((value, weight, bias), expected):
        assert tensor.grad.dtype == dtype
        assert torch.equal(tensor.grad, reference)
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_module_parameters_tied_weight_optimizer_and_state_identity_are_preserved():
    class ExistingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 7)
            self.head = torch.nn.Linear(7, 11)
            self.head.weight = self.embedding.weight

        def forward(self, tokens):
            return batch_invariant_linear(self.embedding(tokens), self.head.weight, self.head.bias)

    module = ExistingModule().double()
    original_parameters = dict(module.named_parameters())
    original_state = {name: value.clone() for name, value in module.state_dict().items()}
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    output = module(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    output.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert all(torch.equal(module.state_dict()[name], value) for name, value in original_state.items())
    optimizer.step()
    assert module.head.weight is module.embedding.weight
    assert set(module.state_dict()) == set(original_state)
    assert all(dict(module.named_parameters())[name] is parameter for name, parameter in original_parameters.items())
    assert all(optimizer.param_groups[0]["params"][i] is parameter
               for i, parameter in enumerate(original_parameters.values()))
    assert any(not torch.equal(module.state_dict()[name], value) for name, value in original_state.items())


@pytest.mark.parametrize("case,match", [
    ("tp", "tensor_parallel_size=1"), ("bool_tp", "tensor_parallel_size=1"),
    ("scalar", "at least one dimension"), ("weight_rank", "at least one dimension"),
    ("features", "feature dimensions"), ("bias", "bias must have shape"),
    ("dtype", "matching devices and dtypes"), ("integer", "real floating point"),
])
def test_unsupported_contract_fails_before_computation(case, match):
    value, weight, bias = torch.randn(3, 5), torch.randn(7, 5), torch.randn(7)
    kwargs = {}
    if case == "tp": kwargs["tensor_parallel_size"] = 2
    if case == "bool_tp": kwargs["tensor_parallel_size"] = True
    if case == "scalar": value = torch.tensor(1.)
    if case == "weight_rank": weight = weight.unsqueeze(0)
    if case == "features": value = torch.randn(3, 6)
    if case == "bias": bias = bias.unsqueeze(0)
    if case == "dtype": weight = weight.double()
    if case == "integer": value, weight, bias = value.long(), weight.long(), bias.long()
    with pytest.raises(ValueError, match=match):
        batch_invariant_linear(value, weight, bias, **kwargs)


@pytest.fixture
def cuda_ieee():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("fixed-tile numerical invariance requires an NVIDIA CUDA device")
    pytest.importorskip("triton")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cuda_row_counts_chunking_duplicates_and_permutations_are_bit_exact(cuda_ieee, dtype, with_bias):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable on this GPU")
    torch.manual_seed(749)
    # Deliberate input/weight/bias strides and non-multiple K/N tails.
    value = torch.randn(128, 70, device="cuda", dtype=dtype)[:, ::2]
    weight = torch.randn(35, 67, device="cuda", dtype=dtype).T
    bias = torch.randn(134, device="cuda", dtype=dtype)[::2] if with_bias else None
    full = batch_invariant_linear(value, weight, bias)
    for rows in (1, 16, 32, 33, 64, 65, 128):
        assert torch.equal(batch_invariant_linear(value[:rows], weight, bias), full[:rows]), rows
    prefix_and_decode = torch.cat([batch_invariant_linear(value[:60], weight, bias)] + [
        batch_invariant_linear(value[i:i + 1], weight, bias) for i in range(60, 64)
    ])
    assert torch.equal(prefix_and_decode, full[:64])
    order = torch.randperm(128, device="cuda")
    assert torch.equal(batch_invariant_linear(value[order], weight, bias), full[order])
    duplicates = torch.tensor([0, 63, 63, 64, 127, 0, 31, 31, 31], device="cuda")
    assert torch.equal(batch_invariant_linear(value[duplicates], weight, bias), full[duplicates])
    assert torch.equal(batch_invariant_linear(value.reshape(2, 64, 35), weight, bias).reshape(128, 67), full)
    # A FP64 reference checks the operation separately from exact shape invariance.
    reference = F.linear(value.double(), weight.double(), None if bias is None else bias.double())
    epsilon = torch.finfo(dtype).eps
    torch.testing.assert_close(full.double(), reference, rtol=epsilon, atol=epsilon / 8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dimensions,k", [((2048, 1024, 1024), 1024), ((3072, 3072), 1024), ((33, 65, 7), 37)])
def test_cuda_packed_and_separate_weight_projections_are_bit_exact(cuda_ieee, dtype, dimensions, k):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable on this GPU")
    torch.manual_seed(523)
    value = torch.randn(64, k, device="cuda", dtype=dtype)
    weights = [torch.randn(n, k, device="cuda", dtype=dtype) for n in dimensions]
    biases = [torch.randn(n, device="cuda", dtype=dtype) for n in dimensions]
    for bias_values in (None, biases):
        packed = batch_invariant_linear(value, torch.cat(weights), None if bias_values is None else torch.cat(bias_values))
        separate = torch.cat([batch_invariant_linear(value, weight, None if bias_values is None else bias_values[i])
                              for i, weight in enumerate(weights)], dim=-1)
        assert torch.equal(packed, separate)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_backward_matches_fp32_reference_and_rejects_tf32(cuda_ieee, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable on this GPU")
    value = torch.randn(2, 33, 35, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(67, 35, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(67, device="cuda", dtype=dtype, requires_grad=True)
    output = batch_invariant_linear(value, weight, bias)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    g = gradient.reshape(-1, 67).float()
    references = ((g @ weight.detach().float()).reshape_as(value).to(dtype),
                  (g.T @ value.detach().reshape(-1, 35).float()).to(dtype), g.sum(0).to(dtype))
    for tensor, reference in zip((value, weight, bias), references):
        assert torch.equal(tensor.grad, reference)
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(RuntimeError, match="matmul.allow_tf32=False"):
        batch_invariant_linear(value, weight, bias).sum().backward()


def test_cuda_float32_is_rejected_instead_of_using_tf32(cuda_ieee):
    with pytest.raises(ValueError, match="requires BF16 or FP16"):
        batch_invariant_linear(torch.randn(3, 5, device="cuda"), torch.randn(7, 5, device="cuda"))


@pytest.mark.parametrize("shape,features", [((0, 5), 7), ((3, 0), 7), ((3, 5), 0)])
def test_cuda_empty_dimensions(cuda_ieee, shape, features):
    value = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(features, shape[-1], device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(features, device="cuda", dtype=torch.float16, requires_grad=True)
    output = batch_invariant_linear(value, weight, bias)
    assert torch.equal(output, F.linear(value, weight, bias))
    output.sum().backward()
    assert all(t.grad is not None and t.grad.shape == t.shape for t in (value, weight, bias))

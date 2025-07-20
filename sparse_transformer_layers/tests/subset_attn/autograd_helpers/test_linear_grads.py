import pytest
import torch

from sparse_transformer_layers.subset_attn.autograd_helpers import linear_grads


@pytest.fixture
def setup_tensors(device):
    """Create tensors for testing both stacked and non-stacked modes."""
    # Set seed for reproducibility
    torch.manual_seed(1)

    batch_size = 8
    in_features = 16
    out_features = 32

    # Regular tensors
    inputs = torch.randn(batch_size, in_features, device=device)
    grad_output = torch.randn(batch_size, out_features, device=device)

    # Stacked tensors (for k and v)
    num_projections = 2
    stacked_grad_output = torch.randn(
        num_projections, batch_size, out_features, device=device
    )

    return {
        "inputs": inputs,
        "grad_output": grad_output,
        "stacked_grad_output": stacked_grad_output,
        "shapes": {
            "batch_size": batch_size,
            "in_features": in_features,
            "out_features": out_features,
            "num_projections": num_projections,
        },
    }


@pytest.mark.cuda_if_available
def test_none_grad_output(setup_tensors):
    """Test handling of None grad_output."""
    tensors = setup_tensors
    inputs = tensors["inputs"]

    weight_grad, bias_grad = linear_grads(None, inputs, True, True)

    assert weight_grad is None
    assert bias_grad is None


# === Non-stacked mode tests ===


@pytest.mark.cuda_if_available
def test_non_stacked_both_grads(setup_tensors):
    """Test non-stacked mode with both weight and bias gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    grad_output = tensors["grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(grad_output, inputs, True, True)

    # Check shapes
    assert weight_grad.shape == (shapes["out_features"], shapes["in_features"])
    assert bias_grad.shape == (shapes["out_features"],)

    # Reference implementation for validation
    ones = inputs.new_ones(inputs.size(0), 1)
    augmented_input = torch.cat([inputs, ones], dim=1)
    combined_grad = torch.mm(grad_output.t(), augmented_input)
    expected_weight_grad = combined_grad[:, :-1]
    expected_bias_grad = combined_grad[:, -1]

    assert torch.allclose(weight_grad, expected_weight_grad)
    assert torch.allclose(bias_grad, expected_bias_grad)


@pytest.mark.cuda_if_available
def test_non_stacked_weight_only(setup_tensors):
    """Test non-stacked mode with only weight gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    grad_output = tensors["grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(grad_output, inputs, True, False)

    # Check results
    assert weight_grad.shape == (shapes["out_features"], shapes["in_features"])
    assert bias_grad is None

    # Validate against reference implementation
    expected_weight_grad = torch.mm(grad_output.t(), inputs)
    assert torch.allclose(weight_grad, expected_weight_grad)


@pytest.mark.cuda_if_available
def test_non_stacked_bias_only(setup_tensors):
    """Test non-stacked mode with only bias gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    grad_output = tensors["grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(grad_output, inputs, False, True)

    # Check results
    assert weight_grad is None
    assert bias_grad.shape == (shapes["out_features"],)

    # Validate against reference implementation
    expected_bias_grad = grad_output.sum(0)
    assert torch.allclose(bias_grad, expected_bias_grad)


@pytest.mark.cuda_if_available
def test_non_stacked_no_grads(setup_tensors):
    """Test non-stacked mode with no gradients needed."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    grad_output = tensors["grad_output"]

    weight_grad, bias_grad = linear_grads(grad_output, inputs, False, False)

    assert weight_grad is None
    assert bias_grad is None


# === Stacked mode tests ===


@pytest.mark.cuda_if_available
def test_stacked_both_grads(setup_tensors):
    """Test stacked mode with both weight and bias gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    stacked_grad_output = tensors["stacked_grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(stacked_grad_output, inputs, True, True)

    # Check shapes
    assert weight_grad.shape == (
        shapes["num_projections"],
        shapes["out_features"],
        shapes["in_features"],
    )
    assert bias_grad.shape == (shapes["num_projections"], shapes["out_features"])

    # Reference implementation for validation
    ones = inputs.new_ones(inputs.size(0), 1)
    augmented_input = torch.cat([inputs, ones], dim=1)

    expected_results = []
    for i in range(shapes["num_projections"]):
        grad_i = stacked_grad_output[i]
        combined_i = torch.mm(grad_i.t(), augmented_input)
        expected_results.append(combined_i)
    expected_combined_grad = torch.stack(expected_results)

    expected_weight_grad = expected_combined_grad[..., :-1]
    expected_bias_grad = expected_combined_grad[..., -1]

    assert torch.allclose(weight_grad, expected_weight_grad)
    assert torch.allclose(bias_grad, expected_bias_grad)


@pytest.mark.cuda_if_available
def test_stacked_weight_only(setup_tensors):
    """Test stacked mode with only weight gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    stacked_grad_output = tensors["stacked_grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(stacked_grad_output, inputs, True, False)

    # Check results
    assert weight_grad.shape == (
        shapes["num_projections"],
        shapes["out_features"],
        shapes["in_features"],
    )
    assert bias_grad is None

    # Validate
    expected_grads = []
    for i in range(shapes["num_projections"]):
        grad_i = stacked_grad_output[i]
        expected_i = torch.mm(grad_i.t(), inputs)
        expected_grads.append(expected_i)
    expected_weight_grad = torch.stack(expected_grads)

    assert torch.allclose(weight_grad, expected_weight_grad)


@pytest.mark.cuda_if_available
def test_stacked_bias_only(setup_tensors):
    """Test stacked mode with only bias gradients."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    stacked_grad_output = tensors["stacked_grad_output"]
    shapes = tensors["shapes"]

    weight_grad, bias_grad = linear_grads(stacked_grad_output, inputs, False, True)

    # Check results
    assert weight_grad is None
    assert bias_grad.shape == (shapes["num_projections"], shapes["out_features"])

    # Validate
    expected_bias_grad = stacked_grad_output.sum(1)
    assert torch.allclose(bias_grad, expected_bias_grad)


@pytest.mark.cuda_if_available
def test_stacked_no_grads(setup_tensors):
    """Test stacked mode with no gradients needed."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    stacked_grad_output = tensors["stacked_grad_output"]

    weight_grad, bias_grad = linear_grads(stacked_grad_output, inputs, False, False)

    assert weight_grad is None
    assert bias_grad is None


@pytest.mark.cuda_if_available
def test_bias_trick_consistency(setup_tensors):
    """Test that the bias trick gives the same results as computing separately."""
    tensors = setup_tensors
    inputs = tensors["inputs"]
    grad_output = tensors["grad_output"]

    # Using bias trick (combined computation)
    weight_grad_combined, bias_grad_combined = linear_grads(
        grad_output, inputs, True, True
    )

    # Separate computations
    weight_grad_separate, _ = linear_grads(grad_output, inputs, True, False)
    _, bias_grad_separate = linear_grads(grad_output, inputs, False, True)

    # They should produce the same results
    assert torch.allclose(weight_grad_combined, weight_grad_separate)
    assert torch.allclose(bias_grad_combined, bias_grad_separate)

    # Same test for stacked mode
    stacked_grad = tensors["stacked_grad_output"]

    weight_grad_combined, bias_grad_combined = linear_grads(
        stacked_grad, inputs, True, True
    )

    weight_grad_separate, _ = linear_grads(stacked_grad, inputs, True, False)
    _, bias_grad_separate = linear_grads(stacked_grad, inputs, False, True)

    assert torch.allclose(weight_grad_combined, weight_grad_separate)
    assert torch.allclose(bias_grad_combined, bias_grad_separate)

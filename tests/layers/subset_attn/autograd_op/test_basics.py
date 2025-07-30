from typing import Any, Literal

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from torch import Tensor

from sparse_transformer_layers.layers.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import attention_inputs
from ..conftest import (
    DIFFERENTIABLE_TENSOR_NAMES,
    ordered_autograd_inputs,
    set_requires_grad,
    simple_attention_input_configs,
)


def grad_not_none(inputs: dict[str, Any], name: str, pass_if_none: bool = False):
    if inputs[name] is not None:
        return inputs[name].grad is not None
    return pass_if_none


def grad_same_shape(inputs: dict[str, Any], name: str, pass_if_none: bool = False):
    if inputs[name] is not None and inputs[name].grad is not None:
        return inputs[name].shape == inputs[name].grad.shape
    return pass_if_none


@pytest.mark.cuda_if_available
class TestBasicForwardBackward:
    def test_attention_forward_shape(self, device: str):
        """Test that the forward pass produces output with the correct shape."""
        inputs = attention_inputs(device=device)
        metadata = inputs["metadata"]

        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        expected_shape = (sum(metadata["n_queries"]), metadata["embed_dim"])
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_attention_forward_with_unspecified_keys(
        self,
        device: str,
    ):
        """Test forward pass with queries having all keys unspecified."""
        inputs = attention_inputs(
            n_queries=4, unspecified_query_indices=[0, 2], device=device
        )

        output = GatherAndSubsetAttentionFunction.apply(
            *(ordered_autograd_inputs(inputs))
        )
        assert isinstance(output, Tensor)

        # Check that queries with all keys unspecified produce finite values
        unspecified_indices = inputs["metadata"]["unspecified_query_indices"]
        if unspecified_indices is not None:
            assert not torch.isnan(
                output[unspecified_indices]
            ).any(), "Output for queries with all keys unspecified contains NaN values"

    def test_attention_forward_backward(self, device):
        """Test both forward and backward passes with gradients."""
        inputs = attention_inputs(device=device)
        metadata = inputs["metadata"]

        # Ensure tensors require gradients
        inputs = set_requires_grad(inputs, DIFFERENTIABLE_TENSOR_NAMES)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        # Check output shape
        expected_shape = (sum(metadata["n_queries"]), metadata["embed_dim"])
        assert output.shape == expected_shape

        # Create a simple loss and run backward
        loss = output.sum()
        loss.backward()

        # Check that gradients were properly computed
        assert grad_not_none(inputs, "query_tensor")
        assert grad_not_none(inputs, "sparse_tensor_values")
        assert grad_not_none(inputs, "key_weight")
        assert grad_not_none(inputs, "value_weight")

        assert grad_not_none(inputs, "key_bias")
        assert grad_not_none(inputs, "value_bias")

        # Check gradient shapes
        assert grad_same_shape(inputs, "query_tensor")
        assert grad_same_shape(inputs, "sparse_tensor_values")
        assert grad_same_shape(inputs, "key_weight")
        assert grad_same_shape(inputs, "value_weight")

        assert grad_same_shape(inputs, "key_bias")
        assert grad_same_shape(inputs, "value_bias")

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    @pytest.mark.cuda_if_available
    def test_attention_with_different_rope_settings(
        self,
        device,
        use_rope: Literal["none"] | Literal["precomputed"] | Literal["from_freqs"],
    ) -> None:
        """Test attention with different RoPE settings."""
        inputs = attention_inputs(use_rope=use_rope, device=device)

        # Ensure tensors require gradients
        inputs = set_requires_grad(inputs, DIFFERENTIABLE_TENSOR_NAMES)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients were properly computed
        assert grad_not_none(inputs, "query_tensor")
        assert grad_not_none(inputs, "sparse_tensor_values")
        assert grad_not_none(inputs, "key_weight")
        assert grad_not_none(inputs, "value_weight")

        assert grad_not_none(inputs, "key_bias")
        assert grad_not_none(inputs, "value_bias")

        assert grad_not_none(inputs, "key_rope_encoding", use_rope != "precomputed")
        assert grad_not_none(inputs, "key_positions", use_rope != "from_freqs")
        assert grad_not_none(inputs, "rope_freqs", use_rope != "from_freqs")

        # Check gradient shapes
        assert grad_same_shape(inputs, "query_tensor")
        assert grad_same_shape(inputs, "sparse_tensor_values")
        assert grad_same_shape(inputs, "key_weight")
        assert grad_same_shape(inputs, "value_weight")

        assert grad_same_shape(inputs, "key_bias")
        assert grad_same_shape(inputs, "value_bias")

        assert grad_same_shape(inputs, "key_rope_encoding", use_rope != "precomputed")
        assert grad_same_shape(inputs, "key_positions", use_rope != "from_freqs")
        assert grad_same_shape(inputs, "rope_freqs", use_rope != "from_freqs")

    @pytest.mark.parametrize("tensor_requiring_grads", DIFFERENTIABLE_TENSOR_NAMES)
    @pytest.mark.cuda_if_available
    def test_individual_tensors_forward_backward(
        self,
        device,
        tensor_requiring_grads,
    ) -> None:
        """Test attention with different RoPE settings."""
        if tensor_requiring_grads == "key_rope_encoding":
            use_rope = "precomputed"
        elif tensor_requiring_grads in ("key_positions", "rope_freqs"):
            use_rope = "from_freqs"
        else:
            use_rope = "none"

        if tensor_requiring_grads == "selection_fill":
            use_selection_fill = True
        else:
            use_selection_fill = False

        inputs = attention_inputs(
            use_rope=use_rope, use_selection_fill=use_selection_fill, device=device
        )
        inputs = set_requires_grad(inputs, tensor_requiring_grads)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert grad_not_none(inputs, tensor_requiring_grads)
        assert grad_same_shape(inputs, tensor_requiring_grads)

    # Property-based version using Hypothesis

    @settings(deadline=None)
    @given(tensor_configs=simple_attention_input_configs())
    def test_hypothesis_forward_backward(
        self,
        device,
        tensor_configs: dict,
    ):
        """Hypothesis-based test to try random combinations of inputs"""
        use_biases = tensor_configs["use_biases"]
        use_rope = tensor_configs["use_rope"]
        use_selection_fill = tensor_configs["use_selection_fill"]
        tensors_requiring_grads = tensor_configs["tensors_requiring_grads"]

        inputs = attention_inputs(
            use_biases=use_biases,
            use_rope=use_rope,
            use_selection_fill=use_selection_fill,
            device=device,
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        # Backward pass
        loss = output.sum()
        loss.backward()

        for tensor_name in tensors_requiring_grads:
            assert grad_not_none(inputs, tensor_name)
            assert grad_same_shape(inputs, tensor_name)


@pytest.mark.cuda_if_available
class TestQueryMask:
    @given(
        tensor_configs=simple_attention_input_configs(),
        query_mask_prob=st.floats(0.0, 1.0),
    )
    @settings(max_examples=10, deadline=None)
    def test_query_mask_basic(
        self, device: str, tensor_configs: dict, query_mask_prob: float
    ):
        """Basic test of the query mask in isolation."""
        inputs = attention_inputs(
            **tensor_configs, device=device, query_mask_rate=query_mask_prob
        )

        query_mask = inputs["query_mask"]
        if query_mask_prob > 0.0:
            assert isinstance(query_mask, Tensor)
        else:
            assert query_mask is None
            assume(False)

        # Set query tensor to require grads
        inputs["query_tensor"].requires_grad_(True)

        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )
        assert isinstance(output, Tensor)

        # Test output properly masked (no other masking used)
        assert (output[query_mask] == 0.0).all()
        if output[~query_mask].numel() > 0:
            assert (output[~query_mask] != 0.0).any()

        # Test gradients
        loss = output.sum()
        loss.backward()

        assert inputs["query_tensor"].grad is not None
        assert (inputs["query_tensor"].grad[query_mask] == 0.0).all()
        if inputs["query_tensor"].grad[~query_mask].numel() > 0:
            assert (inputs["query_tensor"].grad[~query_mask] != 0.0).any()


@pytest.mark.cuda_if_available
class TestDropout:
    def test_dropout_effect_on_output(self, device: str):
        """Test that dropout has a measurable effect during training."""
        seed = 1
        # Get inputs without dropout
        inputs_no_dropout = attention_inputs(device=device, dropout_p=0.0, seed=seed)
        # Same inputs with high dropout
        inputs_dropout = attention_inputs(device=device, dropout_p=0.5, seed=seed)

        inputs_no_dropout = ordered_autograd_inputs(inputs_no_dropout)
        inputs_dropout = ordered_autograd_inputs(inputs_dropout)

        # ensure tensor inputs are the same
        for inp_no, inp_with in zip(inputs_no_dropout, inputs_dropout):
            if isinstance(inp_no, Tensor):
                assert torch.equal(inp_no, inp_with)

        # Run forward passes
        output_no_dropout = GatherAndSubsetAttentionFunction.apply(*inputs_no_dropout)
        output_dropout = GatherAndSubsetAttentionFunction.apply(*inputs_dropout)
        assert isinstance(output_no_dropout, Tensor) and isinstance(
            output_dropout, Tensor
        )

        # Outputs should be different when dropout is applied
        assert not torch.allclose(
            output_no_dropout, output_dropout, rtol=1e-4, atol=1e-4
        )

    def test_dropout_training_vs_eval(self, device: str):
        """Test that dropout is only applied in training mode."""
        seed = 1

        # Get inputs with dropout in training mode
        inputs_training = attention_inputs(
            device=device, dropout_p=0.5, training=True, seed=seed
        )
        ordered_inputs_training = ordered_autograd_inputs(inputs_training)

        # Get inputs with dropout in eval mode
        # (same seed so input tensors should be the same)
        inputs_eval = attention_inputs(
            device=device, dropout_p=0.5, training=False, seed=seed
        )
        ordered_inputs_eval = ordered_autograd_inputs(inputs_eval)

        # Double check input tensors are the same with the same generation seed
        for inp_train, inp_eval in zip(ordered_inputs_training, ordered_inputs_eval):
            if isinstance(inp_train, Tensor):
                assert torch.equal(inp_train, inp_eval)

        # Run forward passes
        torch.manual_seed(seed)
        output_training_1 = GatherAndSubsetAttentionFunction.apply(
            *ordered_inputs_training
        )
        assert isinstance(output_training_1, Tensor)

        # If we run again in training mode, should get different results
        torch.manual_seed(seed + 1)  # Different seed
        output_training_2 = GatherAndSubsetAttentionFunction.apply(
            *ordered_inputs_training
        )
        assert isinstance(output_training_2, Tensor)

        # In eval mode, dropout should be ignored
        output_eval = GatherAndSubsetAttentionFunction.apply(*ordered_inputs_eval)
        assert isinstance(output_eval, Tensor)

        # Training outputs should differ from each other
        assert not torch.allclose(
            output_training_1, output_training_2, rtol=1e-4, atol=1e-4
        )

        # Eval mode outputs should be deterministic regardless of dropout_p
        # They should match outputs with dropout_p=0
        inputs_no_dropout = attention_inputs(
            device=device, dropout_p=0.0, training=False, seed=seed
        )
        output_no_dropout = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs_no_dropout)
        )
        assert isinstance(output_no_dropout, Tensor)
        assert torch.allclose(output_eval, output_no_dropout, rtol=1e-4, atol=1e-4)

    def test_dropout_reproducibility(self, device: str):
        """Test that dropout is reproducible with the same seed."""
        seed = 42
        dropout_p = 0.3

        # First run with seed
        inputs1 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed)
        output1 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs1)
        )
        assert isinstance(output1, Tensor)

        # Second run with same seed
        inputs2 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed)
        output2 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs2)
        )
        assert isinstance(output2, Tensor)

        # Outputs should be identical with same seed
        assert torch.allclose(output1, output2)

        # Different seed should give different output
        inputs3 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed + 1)
        output3 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs3)
        )
        assert isinstance(output3, Tensor)

        # Outputs should differ with different seed
        assert not torch.allclose(output1, output3)


@pytest.mark.cuda_if_available
class TestGradcheck:
    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_basic_gradcheck(
        self,
        device,
        use_rope: Literal["none"] | Literal["precomputed"] | Literal["from_freqs"],
    ) -> None:
        """Test gradcheck with different RoPE settings."""
        inputs = attention_inputs(
            use_rope=use_rope, device=device, dtype=torch.double, dropout_p=0.0
        )

        tensors_to_diff = [
            name for name in DIFFERENTIABLE_TENSOR_NAMES if inputs[name] is not None
        ]
        inputs = set_requires_grad(inputs, tensors_to_diff)
        inputs = ordered_autograd_inputs(inputs)

        assert torch.autograd.gradcheck(
            GatherAndSubsetAttentionFunction.apply,  # pyright: ignore[reportArgumentType]
            inputs,
        )

    # Property-based version using hypothesis

    @settings(deadline=None, max_examples=15)
    @given(tensor_configs=simple_attention_input_configs())
    def test_hypothesis_gradcheck(self, device: str, tensor_configs: dict):
        """Hypothesis-based test to try random combinations of inputs"""
        use_biases = tensor_configs["use_biases"]
        use_rope = tensor_configs["use_rope"]
        tensors_requiring_grads = tensor_configs["tensors_requiring_grads"]

        inputs = attention_inputs(
            use_biases=use_biases,
            use_rope=use_rope,
            device=device,
            dtype=torch.double,
            dropout_p=0.0,
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)
        inputs = ordered_autograd_inputs(inputs)

        nondet_tol = 1e-5 if device == "cuda" else 0.0

        # Run gradcheck
        assert torch.autograd.gradcheck(
            GatherAndSubsetAttentionFunction.apply,  # pyright: ignore[reportArgumentType]
            inputs,
            nondet_tol=nondet_tol,
        )

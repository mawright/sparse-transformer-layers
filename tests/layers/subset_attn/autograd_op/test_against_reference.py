from typing import Any, cast, Optional

import pytest
import torch
from hypothesis import given, settings, assume
from torch import Tensor

from pytorch_sparse_utils.batching import padded_to_concatenated
from sparse_transformer_layers.layers.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import (
    attention_inputs,
)
from ..traceable_attn import (
    prep_batched_attention,
    traceable_batched_attention,
    traceable_subset_attention,
)
from ..conftest import (
    exhaustive_attention_input_configs,
    ordered_autograd_inputs,
    set_requires_grad,
)


def compare_intermediates(
    inputs: dict[str, Any],
    subset_outputs: dict[str, Tensor],
    batched_outputs: dict[str, Tensor],
):
    bsz, sparse_height, sparse_width, n_levels, embed_dim = inputs[
        "sparse_tensor"
    ].shape
    n_heads = inputs["metadata"]["n_heads"]
    max_n_queries = max(inputs["metadata"]["n_queries"])
    n_keys_per_query = inputs["metadata"]["n_keys_per_query"]

    # get indexing tuple for going from all keys to keys per query
    key_b, key_i, key_j, key_l = inputs["index_tensor"].unbind(-1)
    key_q = torch.cat(
        [
            torch.arange(q, device=key_b.device)
            .unsqueeze(1)
            .expand(-1, n_keys_per_query)
            for q in inputs["metadata"]["n_queries"]
        ]
    )
    key_pad_mask = subset_outputs["is_specified_mask"].logical_not()

    # Gather keys
    batched_keys = batched_outputs["keys"].view(
        bsz, sparse_height, sparse_width, n_levels, embed_dim
    )
    stacked_keys_from_batched = batched_keys[key_b, key_i, key_j, key_l]
    subset_keys = subset_outputs["keys"].clone()
    stacked_keys_from_batched[key_pad_mask] = 0.0
    subset_keys[key_pad_mask] = 0.0
    subset_keys = subset_keys.reshape_as(stacked_keys_from_batched)
    assert torch.allclose(
        subset_keys,
        stacked_keys_from_batched,
        atol=1e-3,
        rtol=1e-2,
    ), f"max key diff: {(subset_keys - stacked_keys_from_batched).abs().max()}"

    # same for values...
    batched_values = batched_outputs["values"].view(
        bsz, sparse_height, sparse_width, n_levels, embed_dim
    )
    stacked_values_from_batched = batched_values[key_b, key_i, key_j, key_l]
    subset_values = subset_outputs["values"].clone()
    stacked_values_from_batched[key_pad_mask] = 0.0
    subset_values[key_pad_mask] = 0.0
    subset_values = subset_values.reshape_as(stacked_values_from_batched)
    assert torch.allclose(
        subset_values,
        stacked_values_from_batched,
        atol=1e-3,
        rtol=1e-3,
    ), f"max value diff: {(subset_values - stacked_values_from_batched).abs().max()}"

    # attention scores
    batched_attn_scores_bqhwlh = (
        batched_outputs["attn_scores"]
        .permute(0, 2, 3, 1)
        .reshape(bsz, max_n_queries, sparse_height, sparse_width, n_levels, n_heads)
    )
    # query, key, head
    stacked_attn_scores_from_batched = batched_attn_scores_bqhwlh[
        key_b, key_q, key_i, key_j, key_l
    ].clone()
    subset_attn_scores = subset_outputs["attn_scores"].transpose(-2, -1).clone()
    stacked_attn_scores_from_batched[key_pad_mask] = 0.0
    subset_attn_scores[key_pad_mask] = 0.0
    assert torch.allclose(
        subset_attn_scores,
        stacked_attn_scores_from_batched,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "max scores diff: "
        f"{(subset_attn_scores - stacked_attn_scores_from_batched).abs().max()}"
    )

    # masked attention scores
    batched_attn_scores_masked_bqhwlh = (
        batched_outputs["attn_scores_masked"]
        .permute(0, 2, 3, 1)
        .reshape(bsz, max_n_queries, sparse_height, sparse_width, n_levels, n_heads)
    )
    # query, key, head
    stacked_attn_scores_masked_from_batched = batched_attn_scores_masked_bqhwlh[
        key_b, key_q, key_i, key_j, key_l
    ]
    subset_attn_scores_masked = subset_outputs["attn_scores_masked"].transpose(-1, -2)
    assert torch.allclose(
        subset_attn_scores_masked,
        stacked_attn_scores_masked_from_batched,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "max scores masked diff:"
        f"{(subset_attn_scores_masked - stacked_attn_scores_masked_from_batched.abs().max())}"
    )

    ### check that the indices are the same between index tensor and attn mask??
    # batched_attn_mask_bqhwl = batched_outputs["attn_mask"].view(
    #     bsz, n_queries, sparse_height, sparse_width, n_levels
    # )
    # nonmask_indices = batched_attn_mask_bqhwl.logical_not().nonzero()

    # batched_index_tensor = inputs["batched_index_tensor"]
    # batched_index_tensor_bqijl = (
    #     torch.arange(n_queries, device=batched_index_tensor.device)
    #     .view(1, n_queries, 1, 1)
    #     .expand(bsz, -1, n_keys_per_query, 1)
    #     .contiguous()
    # )
    # batched_index_tensor_bqijl[(batched_index_tensor == -1).all(-1, keepdim=True)] = -1
    # batched_index_tensor_bqijl = batched_index_tensor_bqijl.expand(
    #     -1, -1, -1, 5
    # ).contiguous()
    # batched_index_tensor_bqijl[..., 0] = batched_index_tensor[..., 0]
    # batched_index_tensor_bqijl[..., 2:] = batched_index_tensor[..., 1:]
    # index_tensor_bqijl = remove_batch_dim_and_concat(
    #     batched_index_tensor_bqijl, inputs["query_padding_mask"]
    # )[0]
    # specified_subset_indices = index_tensor_bqijl[inputs["is_specified_mask"]]

    # matched = (specified_subset_indices[None] == nonmask_indices[:, None]).all(-1)

    # attention weights
    batched_attn_weights_bqhwlh = (
        batched_outputs["attn_weights"]
        .permute(0, 2, 3, 1)
        .reshape(bsz, max_n_queries, sparse_height, sparse_width, n_levels, n_heads)
    )
    # query, key, head
    stacked_attn_weights_from_batched = batched_attn_weights_bqhwlh[
        key_b, key_q, key_i, key_j, key_l
    ]
    subset_attn_weights = subset_outputs["attn_weights"].transpose(-1, -2)
    assert torch.allclose(
        subset_attn_weights, stacked_attn_weights_from_batched, atol=1e-6, rtol=1e-3
    ), (
        "max attn_weight difference: "
        f"{(subset_attn_weights - stacked_attn_weights_from_batched).abs().max()}"
    )


@pytest.mark.cuda_if_available
class TestAgainstReferenceUnit:
    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_forward_against_traceable_stacked(self, device: str, use_rope) -> None:
        """Basic test of the forward method against a reference implementation
        that doesn't have optimizations."""
        inputs = attention_inputs(use_rope=use_rope, device=device, dropout_p=0.0)
        inputs = ordered_autograd_inputs(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*inputs)
        reference_output = traceable_subset_attention(
            *inputs, batch_kv_projection=False
        )
        assert isinstance(optimized_output, Tensor)
        assert isinstance(reference_output, Tensor)

        abs_difference = torch.abs(optimized_output - reference_output)
        print(f"Biggest absolute difference: {abs_difference.max().item()}")
        assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

        # Test again while letting the subset version use the input projection
        # batching optimization
        reference_output_2 = traceable_subset_attention(*inputs)
        assert isinstance(reference_output_2, Tensor)

        assert torch.allclose(
            optimized_output, reference_output_2, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_forward_against_traceable_batched(self, device: str, use_rope) -> None:
        """Basic test of the forward method against a reference implementation that
        uses padding instead of stacking the queries, and masking instead of key
        subsets
        """
        inputs = attention_inputs(
            use_rope=use_rope,
            device=device,
            dropout_p=0.0,
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs)
        assert isinstance(optimized_output, Tensor)

        batched_reference_output = traceable_batched_attention(**batched_inputs)
        stacked_reference_output = padded_to_concatenated(
            batched_reference_output, inputs["query_padding_mask"]
        )[0]
        assert isinstance(stacked_reference_output, Tensor)

        assert torch.allclose(
            optimized_output, stacked_reference_output, atol=1e-6, rtol=1e-4
        ), (
            "max_difference: "
            f"{(optimized_output - stacked_reference_output).abs().max()}"
        )

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_traceables_against_each_other(
        self,
        device: str,
        use_rope,
    ) -> None:
        """Test equivalence of the two traceable implementations."""
        inputs = attention_inputs(
            use_rope=use_rope, device=device, dropout_p=0.0, training=False
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        subset_output = traceable_subset_attention(
            *ordered_inputs, return_extended_outputs=True
        )
        batched_output = traceable_batched_attention(
            **batched_inputs, return_extended_outputs=True
        )

        # check equality of intermediate values
        compare_intermediates(inputs, subset_output, batched_output)  # type: ignore

        subset_attn_out: Tensor = subset_output["attn_output"]  # type: ignore
        batched_attn_out: Tensor = batched_output["attn_output"]  # type: ignore

        batched_attn_out_stacked = padded_to_concatenated(
            batched_attn_out, inputs["query_padding_mask"]
        )[0]

        assert subset_attn_out.shape == batched_attn_out_stacked.shape

        abs_difference = torch.abs(subset_attn_out - batched_attn_out_stacked)
        print(f"Biggest absolute difference: {abs_difference.max().item()}")

        assert torch.allclose(subset_attn_out, batched_attn_out_stacked, atol=1e-6)


@pytest.mark.cuda_if_available
class TestAgainstReferenceHypothesis:
    @settings(deadline=None)
    @given(
        inputs_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64]
        )
    )
    def test_forward_against_traceable_stacked(
        self,
        inputs_config: dict[str, Any],
        device: str,
    ) -> None:
        """Property-based forward test against the reference."""
        inputs = attention_inputs(
            **inputs_config, training=False, dropout_p=0.0, device=device
        )
        inputs = ordered_autograd_inputs(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*inputs)
        reference_output = traceable_subset_attention(
            *inputs, batch_kv_projection=False
        )
        assert isinstance(optimized_output, Tensor)
        assert isinstance(reference_output, Tensor)

        assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

        # Test again while letting the subset version use the input projection
        # batching optimization
        reference_output_2 = traceable_subset_attention(*inputs)
        assert isinstance(reference_output_2, Tensor)

        assert torch.allclose(
            optimized_output, reference_output_2, rtol=1e-4, atol=1e-5
        )

    @settings(deadline=None)
    @given(
        inputs_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64]
        )
    )
    def test_forward_against_traceable_batched(
        self, inputs_config: dict[str, Any], device: str
    ) -> None:
        """Basic test of the forward method against a reference implementation that
        uses padding instead of stacking the queries, and masking instead of key
        subsets
        """
        assume(not inputs_config["use_selection_fill"])  # not implemented in batched
        inputs = attention_inputs(
            **inputs_config,
            dropout_p=0.0,
            training=False,
            device=device,
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs)
        assert isinstance(optimized_output, Tensor)
        batched_reference_output = traceable_batched_attention(**batched_inputs)

        stacked_reference_output: Tensor = padded_to_concatenated(
            batched_reference_output, inputs["query_padding_mask"]
        )[0]

        if inputs_config["dtype"] == torch.float32:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5  # default
        assert torch.allclose(
            optimized_output, stacked_reference_output, atol=atol, rtol=rtol
        ), (
            "max_difference: "
            f"{(optimized_output - stacked_reference_output).abs().max()}"
        )

    @settings(deadline=None)
    @given(
        inputs_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64]
        )
    )
    def test_traceables_against_each_other(
        self,
        inputs_config: dict[str, Any],
        device: str,
    ) -> None:
        """Test equivalence of the two traceable implementations."""
        assume(not inputs_config["use_selection_fill"])  # not implemented in batched
        inputs = attention_inputs(
            **inputs_config, device=device, dropout_p=0.0, training=False
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        subset_output = traceable_subset_attention(
            *ordered_inputs, return_extended_outputs=True
        )
        batched_output = traceable_batched_attention(
            **batched_inputs, return_extended_outputs=True
        )

        # check equality of intermediate values
        compare_intermediates(inputs, subset_output, batched_output)  # type: ignore

        subset_attn_out: Tensor = subset_output["attn_output"]  # type: ignore
        batched_attn_out: Tensor = batched_output["attn_output"]  # type: ignore

        batched_attn_out_stacked = padded_to_concatenated(
            batched_attn_out, inputs["query_padding_mask"]
        )[0]

        assert subset_attn_out.shape == batched_attn_out_stacked.shape

        abs_difference = torch.abs(subset_attn_out - batched_attn_out_stacked)

        assert torch.allclose(
            subset_attn_out, batched_attn_out_stacked, atol=1e-3, rtol=1e-2
        ), f"max output diff: {abs_difference.max()}"


@pytest.mark.cuda_if_available
class TestGradientsHypothesis:

    @settings(deadline=None)
    @given(
        inputs_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64],
            min_requiring_grads=1,
        )
    )
    def test_gradients_against_traceable(
        self, device: str, inputs_config: dict[str, Any]
    ):
        """Test gradients against the reference implementation that uses autograd"""
        tensors_requiring_grads = inputs_config["tensors_requiring_grads"]

        # set up inputs
        inputs = attention_inputs(**inputs_config, device=device, dropout_p=0.0)
        inputs = set_requires_grad(inputs, tensors_requiring_grads)

        # make a fresh copy of input tensors for reference implementation
        inputs_copy = {
            k: (
                v.clone().detach().requires_grad_(v.requires_grad)
                if isinstance(v, Tensor)
                else v
            )
            for k, v in inputs.items()
        }
        ordered_optimized_inputs = ordered_autograd_inputs(inputs)
        ordered_reference_inputs = ordered_autograd_inputs(inputs_copy)

        torch.manual_seed(inputs_config["seed"])

        # get outputs
        optimized_output = GatherAndSubsetAttentionFunction.apply(
            *ordered_optimized_inputs
        )
        assert isinstance(optimized_output, Tensor)
        reference_output = traceable_subset_attention(
            *ordered_reference_inputs, batch_kv_projection=False
        )
        assert isinstance(reference_output, Tensor)

        # check outputs match
        assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

        # Create random gradient for backprop
        grad_output = torch.randn_like(optimized_output)
        optimized_output.backward(grad_output)
        reference_output.backward(grad_output.clone())

        # Compare gradients
        for tensor_name in inputs:
            opt_input = inputs[tensor_name]
            ref_input = inputs_copy[tensor_name]
            if isinstance(opt_input, Tensor) and opt_input.requires_grad:
                assert (
                    opt_input.grad is not None
                ), f"Optimized grad is None for input {tensor_name}"
                assert (
                    ref_input.grad is not None
                ), f"Reference grad is None for input {tensor_name}"

                diff = torch.abs(opt_input.grad - ref_input.grad)

                assert torch.allclose(
                    opt_input.grad, ref_input.grad, rtol=1e-3, atol=1e-4
                ), f"Grad mismatch for input {tensor_name}: diff max={diff.max()} mean={diff.mean()}"

    @settings(deadline=None)
    @given(
        inputs_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64],
            min_requiring_grads=1,
        )
    )
    def test_gradients_against_traceable_batched(
        self, device: str, inputs_config: dict[str, Any]
    ):
        """Test gradients against the reference implementation that uses autograd"""
        assume(not inputs_config["use_selection_fill"])  # not implemented in batched
        tensors_requiring_grads = inputs_config["tensors_requiring_grads"]
        torch.set_anomaly_enabled(True)

        # set up inputs
        inputs = attention_inputs(**inputs_config, device=device, dropout_p=0.0)
        inputs = set_requires_grad(inputs, tensors_requiring_grads)

        # make a fresh copy of input tensors for reference implementation
        reference_inputs = {
            k: (
                v.clone().detach().requires_grad_(v.requires_grad)
                if isinstance(v, Tensor)
                else v
            )
            for k, v in inputs.items()
        }

        optimized_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(reference_inputs)

        torch.manual_seed(inputs_config["seed"])

        # get outputs
        optimized_output = GatherAndSubsetAttentionFunction.apply(*optimized_inputs)
        assert isinstance(optimized_output, Tensor)
        batched_outputs = cast(
            dict[str, Optional[Tensor]],
            traceable_batched_attention(**batched_inputs, return_extended_outputs=True),
        )

        # check outputs match

        stacked_batched_output = padded_to_concatenated(
            batched_outputs["attn_output"], reference_inputs["query_padding_mask"]
        )[0]
        assert torch.allclose(
            optimized_output, stacked_batched_output, rtol=1e-3, atol=1e-4
        ), f"max output diff: {(optimized_output - stacked_batched_output).abs().max()}"

        # Create random gradient for backprop
        grad_output = torch.randn_like(optimized_output)
        optimized_output.backward(grad_output)
        stacked_batched_output.backward(grad_output.clone())

        # Compare gradients
        for k in tensors_requiring_grads:
            ref_input = reference_inputs[k]
            opt_input = inputs[k]
            # if isinstance(ref_input, Tensor) and ref_input.requires_grad:
            assert opt_input.grad is not None, f"Optimized grad is None for {k}"
            assert ref_input.grad is not None, f"Reference grad is None for {k}"

            diff = torch.abs(opt_input.grad - ref_input.grad)

            assert torch.allclose(
                opt_input.grad, ref_input.grad, rtol=1e-2, atol=1e-3
            ), f"Grad mismatch for {k}: diff max={diff.max()} mean={diff.mean()}"

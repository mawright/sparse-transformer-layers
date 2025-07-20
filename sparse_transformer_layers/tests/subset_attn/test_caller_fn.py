from typing import Any

import pytest
import torch
from torch import Tensor
from hypothesis import given, settings
from hypothesis import strategies as st

from sparse_transformer_layers.subset_attn import (
    batch_sparse_index_subset_attn,
    BatchSparseIndexSubsetAttention,
)

from ..constants import (
    EMBED_DIM,
    N_FREQ_GROUPS,
    N_HEADS,
    N_KEYS_PER_QUERY,
    POSITION_DIM,
)
from .conftest import (
    exhaustive_attention_input_configs,
    subset_key_rope_inputs,
    set_requires_grad,
)
from .input_generation import attention_inputs


@pytest.mark.cuda_if_available
@pytest.mark.parametrize(
    "key_pos_encoding_type",
    ["given", "computed", None],
    ids=[
        "key_pos_encoding_type=given",
        "key_pos_encoding_type=computed",
        "key_pos_encoding_type=None",
    ],
)
@pytest.mark.parametrize(
    "scale_factor", [None, 0.5], ids=["scale_factor=None", "scale_factor=0.5"]
)
def test_end_to_end_subset_attn(
    setup_sparse_tensor,
    setup_attention_index_tensor,
    device,
    key_pos_encoding_type,
    scale_factor,
):
    """Test end-to-end gather and subset attention."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_attention_index_tensor

    # Create query vectors
    query_tensor = torch.randn(
        index_tensor.shape[0],
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )

    # Create attention parameters
    key_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    value_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    key_bias = torch.randn(
        EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    value_bias = torch.randn(
        EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    key_pos_encoding = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "given"
        else None
    )
    key_positions = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            POSITION_DIM,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
    )
    rope_freqs = (
        torch.randn(
            POSITION_DIM,
            N_FREQ_GROUPS,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
    )

    # Run the operation
    attended, is_specified_mask = batch_sparse_index_subset_attn(
        sparse_tensor,
        index_tensor,
        query_tensor,
        N_HEADS,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        key_rope_encoding=key_pos_encoding,
        key_positions=key_positions,
        rope_freqs=rope_freqs,
        scale_factor=scale_factor,
    )

    # Check output shapes
    expected_output_shape = list(query_tensor.shape[:-1])
    expected_output_shape.append(EMBED_DIM)
    assert attended.shape == tuple(expected_output_shape)
    assert is_specified_mask.shape == tuple(index_tensor.shape[:-1])

    # Compute loss and check gradient flow
    loss = attended.sum()
    loss.backward()

    assert query_tensor.grad is not None
    assert key_weight.grad is not None
    assert value_weight.grad is not None
    assert key_bias.grad is not None
    assert value_bias.grad is not None


@pytest.mark.cuda_if_available
class TestModule:
    @settings(deadline=None)
    @given(embed_dim=st.integers(min_value=1, max_value=128), use_bias=st.booleans())
    def test_initialization(self, embed_dim: int, use_bias: bool, device: str):
        module = BatchSparseIndexSubsetAttention(embed_dim, use_bias).to(device)

        assert module.kv_proj.weight.device.type == torch.device(device).type
        assert module.kv_proj.in_features == embed_dim
        assert module.kv_proj.out_features == embed_dim * 2

        if use_bias:
            assert module.kv_proj.bias is not None
        else:
            assert module.kv_proj.bias is None

    @settings(deadline=None)
    @given(
        input_config=exhaustive_attention_input_configs(
            dtypes=[torch.float32, torch.float64]
        )
    )
    def test_forward(self, input_config: dict[str, Any], device: str):
        """Test that the module forward pass produces the same behavior as calling
        the wrapper function directly"""
        module = BatchSparseIndexSubsetAttention(
            embed_dim=input_config["embed_dim"],
            use_bias=input_config["use_biases"],
            dtype=input_config["dtype"],
        ).to(device)
        inputs = attention_inputs(**input_config, device=device)
        inputs.update(subset_key_rope_inputs(inputs))

        # Copy module-generated key and value weights into input dict tensors
        module_key_weight, module_value_weight = module.kv_proj.weight.chunk(2, dim=0)
        with torch.no_grad():
            inputs["key_weight"].copy_(module_key_weight)
            inputs["value_weight"].copy_(module_value_weight)
        if input_config["use_biases"]:
            module_key_bias, module_value_bias = module.kv_proj.bias.chunk(2, dim=0)
            with torch.no_grad():
                inputs["key_bias"].copy_(module_key_bias)
                inputs["value_bias"].copy_(module_value_bias)
        else:
            assert module.kv_proj.bias is None
            assert inputs["key_bias"] is None and inputs["value_bias"] is None

        inputs_module = inputs.copy()
        inputs_module = {
            k: v.detach().clone() if isinstance(v, Tensor) else v
            for k, v in inputs_module.items()
        }

        # set requires_grad
        tensors_requiring_grads: list[str] = input_config["tensors_requiring_grads"]
        if "sparse_tensor_values" in tensors_requiring_grads:
            tensors_requiring_grads.remove("sparse_tensor_values")
            tensors_requiring_grads.append("sparse_tensor")
        if len(tensors_requiring_grads) > 0:
            inputs = set_requires_grad(inputs, tensors_requiring_grads)
            inputs_module = set_requires_grad(inputs_module, tensors_requiring_grads)

        # call the forward passes
        wrapper_out, wrapper_is_specified = batch_sparse_index_subset_attn(
            sparse_tensor=inputs["sparse_tensor"],
            index_tensor=inputs["index_tensor"],
            query_tensor=inputs["query_tensor"],
            n_heads=inputs["n_heads"],
            key_weight=inputs["key_weight"],
            value_weight=inputs["value_weight"],
            key_bias=inputs["key_bias"],
            value_bias=inputs["value_bias"],
            background_embedding=inputs["selection_fill"],
            key_rope_encoding=inputs["key_rope_encoding"],
            key_positions=inputs["key_positions"],
            rope_freqs=inputs["rope_freqs"],
            scale_factor=inputs["scale_factor"],
        )

        module_out, module_is_specified = module(
            sparse_tensor=inputs_module["sparse_tensor"],
            index_tensor=inputs_module["index_tensor"],
            query_tensor=inputs_module["query_tensor"],
            n_heads=inputs_module["n_heads"],
            key_rope_encoding=inputs_module["key_rope_encoding"],
            key_positions=inputs_module["key_positions"],
            rope_freqs=inputs_module["rope_freqs"],
            scale_factor=inputs_module["scale_factor"],
            background_embedding=inputs_module["selection_fill"],
        )

        assert torch.allclose(wrapper_out, module_out)
        assert torch.allclose(wrapper_is_specified, module_is_specified)

        if len(tensors_requiring_grads) > 0:
            with torch.no_grad():
                grad_output = torch.randn_like(wrapper_out)
            wrapper_out.backward(grad_output)
            module_out.backward(grad_output)

            module_kv_params = module.kv_params()

            for tensor_name in tensors_requiring_grads:
                wrapper_grad = inputs[tensor_name].grad
                if tensor_name in module_kv_params:
                    if tensor_name == "key_weight":
                        assert module.kv_proj.weight.grad is not None
                        module_grad = module.kv_proj.weight.grad[: module.embed_dim]
                    elif tensor_name == "value_weight":
                        assert module.kv_proj.weight.grad is not None
                        module_grad = module.kv_proj.weight.grad[module.embed_dim :]
                    elif tensor_name == "key_bias":
                        assert module.kv_proj.bias.grad is not None
                        module_grad = module.kv_proj.bias.grad[: module.embed_dim]
                    elif tensor_name == "value_bias":
                        assert module.kv_proj.bias.grad is not None
                        module_grad = module.kv_proj.bias.grad[module.embed_dim :]
                else:
                    module_grad = inputs_module[tensor_name].grad

                assert wrapper_grad is not None
                assert module_grad is not None
                if wrapper_grad.is_sparse:
                    assert module_grad.is_sparse
                    assert torch.allclose(
                        wrapper_grad.coalesce().values(),
                        module_grad.coalesce().values(),
                        atol=1e-6
                    )
                else:
                    assert torch.allclose(wrapper_grad, module_grad, atol=1e-6)

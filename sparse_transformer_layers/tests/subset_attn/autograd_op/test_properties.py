from typing import Any

import pytest
import torch
from torch import Tensor
from hypothesis import given, settings

from sparse_transformer_layers.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import attention_inputs
from ..traceable_attn import traceable_subset_attention
from ..conftest import (
    exhaustive_attention_input_configs,
    ordered_autograd_inputs,
    set_requires_grad,
)


@pytest.mark.cuda_if_available
@settings(deadline=None, max_examples=25)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.double], min_requiring_grads=1
    )
)
def test_gradcheck_exhaustive(device: str, input_params: dict[str, Any]) -> None:
    """Gradcheck test letting Hypothesis really explore the input space.

    Takes a while to run.
    """
    tensors_requiring_grads = input_params["tensors_requiring_grads"]

    inputs = attention_inputs(**input_params, device=device, dropout_p=0.0)

    inputs = set_requires_grad(inputs, tensors_requiring_grads)
    inputs = ordered_autograd_inputs(inputs)

    nondet_tol = 1e-5 if device == "cuda" else 0.0

    assert torch.autograd.gradcheck(
        GatherAndSubsetAttentionFunction.apply,  # pyright: ignore[reportArgumentType]
        inputs,
        nondet_tol=nondet_tol,
    )


@pytest.mark.cuda_if_available
@settings(deadline=None)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.float32, torch.float64]  # TODO fully implement 16-bit correctness
    )
)
def test_forward_against_traceable(device: str, input_params: dict[str, Any]):
    """Test the forward method against a reference implementation that doesn't have
    optimizations."""
    inputs = attention_inputs(**input_params, device=device, dropout_p=0.0)
    inputs = ordered_autograd_inputs(inputs)

    optimized_output = GatherAndSubsetAttentionFunction.apply(*inputs)
    assert isinstance(optimized_output, Tensor)
    reference_output = traceable_subset_attention(*inputs, batch_kv_projection=False)
    assert isinstance(reference_output, Tensor)

    abs_difference = torch.abs(optimized_output - reference_output)
    print(f"Biggest absolute difference: {abs_difference.max().item()}")
    assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

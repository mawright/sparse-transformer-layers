from typing import Any, Literal, Union, Optional

import pytest
import torch
from hypothesis import strategies as st
from torch import Tensor

from ..constants import (
    ALWAYS_SPECIFIED,
    ALWAYS_UNSPECIFIED,
    BATCH_SIZE,
    N_KEYS_PER_QUERY,
    SPARSE_DIM_1,
    SPARSE_DIM_2,
    SPARSE_DIM_3,
)


DIFFERENTIABLE_TENSOR_NAMES = [
    "query_tensor",
    "sparse_tensor_values",
    "key_weight",
    "value_weight",
    "key_bias",
    "value_bias",
    "selection_fill",
    "key_rope_encoding",
    "key_positions",
    "rope_freqs",
]


def subset_key_rope_inputs(inputs: dict[str, Any]) -> dict[str, Optional[Tensor]]:
    """Go from batched format of the per-key RoPE data (either the encoding vector
    or the positions used to compute the encoding vector) to subset format, by
    extracting the particular keys referred to by the index tensor.
    """
    # shape: sum(n_queries), n_keys_per_query, n_heads, head_dim//2
    key_rope_encoding = (
        inputs["key_rope_encoding"][inputs["index_tensor"].unbind(-1)]
        if inputs["key_rope_encoding"] is not None
        else None
    )

    # shape: sum(n_queries), n_keys_per_query, position_dim
    key_positions = (
        inputs["key_positions"][inputs["index_tensor"].unbind(-1)]
        if inputs["key_positions"] is not None
        else None
    )

    return {"key_rope_encoding": key_rope_encoding, "key_positions": key_positions}


def ordered_autograd_inputs(
    inputs: Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple:
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    subsetted_key_tensors = subset_key_rope_inputs(inputs)

    return (
        inputs["query_tensor"],
        inputs["n_heads"],
        inputs["sparse_tensor_values"],
        inputs["linear_index_tensor"],
        inputs["is_specified_mask"],
        inputs["key_weight"],
        inputs["value_weight"],
        inputs["key_bias"],
        inputs["value_bias"],
        inputs["query_mask"],
        inputs["selection_fill"],
        subsetted_key_tensors["key_rope_encoding"],
        subsetted_key_tensors["key_positions"],
        inputs["rope_freqs"],
        inputs["scale_factor"],
        inputs["dropout_p"],
        inputs["training"],
    )


def set_requires_grad(inputs: dict[str, Any], tensor_names: Union[str, list[str]]):
    """Sets the requires_grad flag to True for specified tensors in the input dict"""
    modified_inputs = inputs.copy()
    if isinstance(tensor_names, str):
        tensor_names = [tensor_names]
    for name in tensor_names:
        if name in modified_inputs and modified_inputs[name] is not None:
            tensor: Tensor = modified_inputs[name].detach().clone()
            modified_inputs[name] = tensor.requires_grad_(True)
    return modified_inputs


def filter_valid_tensor_names(
    use_rope: Union[Literal["none"], Literal["precomputed"], Literal["from_freqs"]],
    use_biases: bool,
    use_selection_fill: bool,
) -> list[str]:
    """Filter tensor names based on the given parameters.

    Returns a list of tensor names that are valid for the given combination
    of use_rope and use_biases parameters.
    """
    # Start with all tensors
    valid_tensors = list(DIFFERENTIABLE_TENSOR_NAMES)

    if use_rope != "precomputed":
        # Remove key_rope_encoding if not using precomputed RoPE
        valid_tensors = [t for t in valid_tensors if t != "key_rope_encoding"]

    if use_rope != "from_freqs":
        # Remove position-based RoPE tensors if not computing RoPE from frequencies
        valid_tensors = [
            t for t in valid_tensors if t not in ["key_positions", "rope_freqs"]
        ]

    if not use_biases:
        # Remove bias tensors if not using biases
        valid_tensors = [
            t for t in valid_tensors if t not in ["key_bias", "value_bias"]
        ]

    if not use_selection_fill:
        valid_tensors = [t for t in valid_tensors if t != "selection_fill"]

    return valid_tensors


def _draw_shared_attention_params(draw, min_requiring_grads: int = 0):
    """Helper function that does the drawing of base parameters for both strategies"""
    use_rope = draw(st.sampled_from(["none", "precomputed", "from_freqs"]))
    use_biases = draw(st.booleans())
    use_selection_fill = draw(st.booleans())

    # Get valid tensor names for these parameters
    available_tensors = filter_valid_tensor_names(
        use_rope, use_biases, use_selection_fill
    )

    # Draw a non-empty subset of available tensors
    tensors_requiring_grads = draw(
        st.lists(
            st.sampled_from(available_tensors),
            min_size=min_requiring_grads,
            max_size=len(available_tensors),
            unique=True,
        )
    )

    # Sample seed
    seed = draw(st.integers(0, int(1e8)))

    return {
        "use_rope": use_rope,
        "use_biases": use_biases,
        "use_selection_fill": use_selection_fill,
        "tensors_requiring_grads": tensors_requiring_grads,
        "seed": seed,
    }


@st.composite
def simple_attention_input_configs(draw, min_requiring_grads: int = 1):
    """Hypothesis strategy for generating valid parameters for attention function
    tests."""
    return _draw_shared_attention_params(draw, min_requiring_grads)


@st.composite
def exhaustive_attention_input_configs(
    draw,
    dtypes: Union[torch.dtype, list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    min_requiring_grads: int = 0,
) -> dict[str, Any]:
    """Strategy that generates all of attention_inputs's input args.

    Args:
        dtypes (Union[torch.dtype, list[torch.dtype]]): Specific dtype or list of
            dtypes to sample from. For gradcheck tests, use torch.double for
            numerical stability. Defaults to both 16-bit floats, float, and double
        min_requiring_grads (int): Minimum number of input tensors to set as
            requiring gradients. For gradcheck tests, this should be at least 1.
            Defaults to 0.
    """
    base_params = _draw_shared_attention_params(draw, min_requiring_grads)

    n_queries = draw(st.integers(min_value=1, max_value=8))
    n_heads = draw(st.integers(min_value=1, max_value=4))

    # Make sure embed_dim is divisible by n_heads
    # and even for compatibility with RoPE
    head_dim = draw(st.integers(min_value=1, max_value=4).map(lambda x: x * 2))
    embed_dim = head_dim * n_heads

    n_keys_per_query = draw(st.integers(min_value=1, max_value=8))
    num_sparse_values = draw(
        st.integers(min_value=max(5, n_keys_per_query), max_value=32)
    )

    # These parameters matter only when use_rope="from_freqs"
    position_dim = draw(st.integers(min_value=1, max_value=4))
    n_freq_groups = draw(st.integers(min_value=1, max_value=position_dim))

    # Parameter for sparse attention
    unspecified_prob = draw(st.floats(min_value=0.0, max_value=0.5))

    # Decide if we want queries with all keys unspecified
    has_unspecified_queries = draw(st.booleans())
    unspecified_query_indices = None
    if has_unspecified_queries and n_queries > 0:
        num_unspecified = draw(st.integers(min_value=0, max_value=min(2, n_queries)))
        if num_unspecified > 0:
            unspecified_query_indices = draw(
                st.lists(
                    st.integers(min_value=0, max_value=n_queries - 1),
                    min_size=1,
                    max_size=num_unspecified,
                    unique=True,
                )
            )

    # Optional probability to no-op certain queries
    use_query_mask = draw(st.booleans())
    if use_query_mask:
        query_mask_rate = draw(st.floats(0.0, 1.0))
    else:
        query_mask_rate = None

    # Sample dtype
    dtypes = [dtypes] if isinstance(dtypes, torch.dtype) else dtypes
    dtype = draw(st.sampled_from(dtypes))

    return {
        "n_queries": n_queries,
        "embed_dim": embed_dim,
        "n_heads": n_heads,
        "n_keys_per_query": n_keys_per_query,
        "num_sparse_values": num_sparse_values,
        "position_dim": position_dim,
        "n_freq_groups": n_freq_groups,
        "unspecified_query_indices": unspecified_query_indices,
        "unspecified_prob": unspecified_prob,
        "query_mask_rate": query_mask_rate,
        "dtype": dtype,
        "use_biases": base_params["use_biases"],
        "use_rope": base_params["use_rope"],
        "use_selection_fill": base_params["use_selection_fill"],
        "tensors_requiring_grads": base_params["tensors_requiring_grads"],
        "seed": base_params["seed"],
    }


@pytest.fixture
def setup_attention_index_tensor(setup_sparse_tensor, device):
    """Create attention index tensor with a mixture of specified and random indices."""
    # Get indices from the sparse tensor and ensure contiguous memory layout
    sparse_indices = setup_sparse_tensor.indices().t().contiguous()

    # Create random regular queries
    n_queries = 2
    index_tensor = torch.zeros(
        n_queries, N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device
    )

    # Assign random batch indices and spatial dimensions
    query_batch_indices = torch.randint(0, BATCH_SIZE, (n_queries,), device=device)
    index_tensor[:, :, 0] = query_batch_indices.unsqueeze(1)
    index_tensor[:, :, 1].random_(0, SPARSE_DIM_1)
    index_tensor[:, :, 2].random_(0, SPARSE_DIM_2)
    index_tensor[:, :, 3].random_(0, SPARSE_DIM_3)

    # Randomly decide which keys will use specified indices (50% probability)
    use_specified = torch.rand(n_queries, N_KEYS_PER_QUERY) < 0.5

    # Pre-compute a dictionary mapping each batch to its sparse indices
    # This avoids repeatedly filtering the sparse tensor for each query
    batch_to_sparse_indices = {
        b: sparse_indices[sparse_indices[:, 0] == b] for b in range(BATCH_SIZE)
    }

    # Replace random indices with specified ones where appropriate
    for q in range(n_queries):
        b = query_batch_indices[q].item()
        batch_specified = batch_to_sparse_indices.get(b, None)  # type: ignore

        # Skip if no specified indices for this batch or no keys to replace
        if batch_specified is None or len(batch_specified) == 0:
            continue

        # Get keys to replace and count
        specified_key_mask = use_specified[q]
        num_specified_keys = int(specified_key_mask.sum().item())

        if num_specified_keys > 0:
            # Sample indices with replacement from the known specified indices
            idx = torch.randint(0, len(batch_specified), (num_specified_keys,))
            index_tensor[q, specified_key_mask, 1:] = batch_specified[idx, 1:]

    # Create test case queries with predefined patterns
    test_queries = []

    # For each batch, create two special test queries
    for b in range(BATCH_SIZE):
        # Query 1: First key is a known specified index, rest are random
        q1 = torch.zeros(N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device)
        q1[:, 0] = b  # Set batch index
        q1[:, 1:].random_(0, max(SPARSE_DIM_1, SPARSE_DIM_2, SPARSE_DIM_3))
        q1[0, 1:] = ALWAYS_SPECIFIED[b, 1:].to(
            device
        )  # First key is the specified point

        # Query 2: All keys point to a known unspecified location
        q2 = torch.zeros(N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device)
        q2[:, 0] = b  # Set batch index
        q2[:, 1:] = (
            ALWAYS_UNSPECIFIED[b, 1:].to(device).unsqueeze(0)
        )  # All keys are unspecified

        test_queries.extend([q1, q2])

    # Combine regular and test queries along the query dimension (dim=0)
    # index_tensor shape: [n_queries, N_KEYS_PER_QUERY, 4]
    # torch.stack(test_queries) shape: [2*BATCH_SIZE, N_KEYS_PER_QUERY, 4]
    combined_tensor = torch.cat([index_tensor, torch.stack(test_queries)], dim=0)
    _, sort_indices = torch.sort(combined_tensor[:, 0, 0])

    return combined_tensor[sort_indices].contiguous()

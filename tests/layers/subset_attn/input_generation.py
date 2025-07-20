from typing import Optional, Union, Literal, Any

import numpy as np
from torch import Tensor
import torch

from pytorch_sparse_utils.indexing.utils import get_sparse_index_mapping
from pytorch_sparse_utils.batching import padded_to_concatenated


def attention_inputs(
    n_queries: Union[int, list[int]] = 4,
    embed_dim: int = 16,
    n_heads: int = 4,
    n_keys_per_query: int = 5,
    use_biases: bool = True,
    use_rope: Union[
        Literal["none"], Literal["precomputed", Literal["from_freqs"]]
    ] = "none",  # none, precomputed, from_freqs
    use_selection_fill: bool = False,
    query_mask_rate: float = 0.0,
    position_dim: int = 2,
    n_freq_groups: int = 1,
    sparse_height: int = 8,
    sparse_width: int = 8,
    sparse_levels: int = 4,
    sparsity: float = 0.9,
    index_hit_rate: float = 0.75,
    unspecified_query_indices: Optional[Union[int, list[int]]] = None,
    use_2d_sparse_features: bool = False,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    dropout_p: float = 0.1,
    training: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[str, Any]:
    """Generate test inputs for sparse attention mechanisms.

    This function creates a comprehensive set of tensors needed for testing
    sparse attention implementations. It supports both batched and stacked formats,
    different forms of positional encodings, and configurable sparsity patterns.

    Args:
        n_queries (Union[int, list[int]]): Number of queries per batch. If an integer,
            creates a single-element batch with that many queries. If a list, each
            element specifies the number of queries for the corresponding batch.
        embed_dim (int): Embedding dimension for queries, keys, and values.
        n_heads (int): Number of attention heads. Must divide embed_dim evenly.
        n_keys_per_query (int): Number of keys each query attends to.
        use_biases (bool): Whether to include bias terms for key and value projections.
        use_rope (Union[Literal["none"], Literal["precomputed"], Literal["from_freqs"]]):
            Rotary positional encoding mode. Options:
            - "none": No positional encoding
            - "precomputed": Use precomputed RoPE values
            - "from_freqs": Generate RoPE from frequency components
        use_selection_fill (bool): Whether to create a `selection_fill` tensor to serve
            as a background embedding for selected keys corresponding to empty pixels
        query_mask_rate (float): If greater than 0, a `query_mask` tensor is created
            with True entries at a rate of query_mask_rate. This tensor is passed to
            the autograd function to selectively turn the neighborhood attention
            operation to a no-op on those queries.
        position_dim (int): Number of dimensions for positional encoding (when using RoPE).
        n_freq_groups (int): Number of frequency groups for RoPE.
        sparse_height (int): Height dimension of the sparse spatial grid.
        sparse_width (int): Width dimension of the sparse spatial grid.
        sparse_levels (int): Number of hierarchical levels in the sparse grid.
        sparsity (float): Target sparsity (proportion of empty entries) in the sparse
            tensor.
        index_hit_rate (float): Proportion of indexed values that actually exist in the
            sparse tensor.
        unspecified_query_indices (Optional[Union[int, list[int]]]): Indices of queries
            that should have no valid keys (all misses). Can be a single index or list
            of indices.
        use_2d_sparse_features (bool): If True, use [n_heads, head_dim] for sparse tensor
            feature dims.
        generate_linear_sparse_tensor_directly (bool): If True, generate linearized
            sparse tensors directly without creating intermediate spatial tensors.
        device (Union[str, torch.device]): Device on which to create tensors.
        dtype (torch.dtype): Data type for tensor values.
        dropout_p (float): Dropout probability for attention.
        training (bool): Whether the model is in training mode.
        seed (Optional[int]): Random seed for reproducibility. Original RNG state is restored after use.
        **kwargs: Additional arguments passed to helper functions.

    Returns:
        dict: A dictionary containing all tensors needed for testing sparse attention:
            - query_tensor: Query tensor in stacked format
            - batched_query_tensor: Query tensor in batched format with padding
            - sparse_tensor: Sparse COO tensor containing key/value features (may be None)
            - index_tensor: Query-to-key mapping indices in stacked format (may be None)
            - batched_index_tensor: Query-to-key mapping indices in batched format
                (may be None)
            - attn_mask_valid_indices: Tensor of shape (n_queries*n_keys_per_query) x 5,
                where each row gives the indices corresponding to a valid (non-masked)
                query-key interaction, the 5 dimensions being
                (batch, query, height, width, level). To be used to create an attn_mask
                tensor for batched attention. For example, Pytorch MultiHeadAttention
                expects an attn_mask where True corresponds to attention interactions
                that should be masked out, so attn_mask_valid_indices designates
                the indices that should be False. Note that F.scaled_dot_product_attention
                in particular expects a tensor where False means to mask out.
            - query_padding_mask: Boolean mask indicating padding in batched tensors
            - query_batch_offsets: Offsets for each batch in stacked format
            - n_heads: Number of attention heads
            - sparse_tensor_values: Values of the sparse tensor
            - linear_index_tensor: Linearized indices for sparse tensor lookup
            - is_specified_mask: Boolean mask indicating valid indices
            - key_weight: Key projection weight matrix
            - value_weight: Value projection weight matrix
            - key_bias: Key projection bias (None if use_biases=False)
            - value_bias: Value projection bias (None if use_biases=False)
            - key_rope_encoding: RoPE encoding for keys in stacked format (if applicable)
            - batched_key_rope_encoding: RoPE encoding for keys in batched format
                (if applicable)
            - key_positions: Key positions for RoPE in stacked format (if applicable)
            - batched_key_positions: Key positions for RoPE in batched format
                (if applicable)
            - rope_freqs: RoPE frequency components (if applicable)
            - selection_fill: Tensor of shape [n_queries, 1, embed_dim] (if applicable)
            - scale_factor: Optional scale factor for attention scores
            - dropout_p: Dropout probability
            - training: Training mode flag
            - metadata: Dictionary with configuration parameters

    Note:
        The function preserves the random state by saving and restoring it when a seed is
            provided.
        The generated tensors are compatible with both batched and stacked computation
            approaches.
    """

    device = torch.device(device)
    if device.type == "cuda":
        torch_rng_state = torch.cuda.get_rng_state(device)
    else:
        torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if isinstance(n_queries, int):
        n_queries = [n_queries]

    # Generate padding mask for converting from batched to stacked format
    query_padding_mask = torch.zeros(
        len(n_queries), max(n_queries), dtype=torch.bool, device=device
    )
    for b, n_queries_b in enumerate(n_queries):
        query_padding_mask[b, n_queries_b:] = True

    query_batch_offsets = torch.tensor(np.cumsum([0] + n_queries), device=device)

    # Generate the sparse tensor inputs
    sparse_tensor, batched_index_tensor = create_sparse_and_index_tensor(
        n_queries=n_queries,
        height=sparse_height,
        width=sparse_width,
        levels=sparse_levels,
        embed_dim=embed_dim,
        n_keys_per_query=n_keys_per_query,
        sparsity=sparsity,
        index_hit_rate=index_hit_rate,
        use_2d_features=use_2d_sparse_features,
        n_heads=n_heads,
        unspecified_query_indices=unspecified_query_indices,
        device=device,
        dtype=dtype,
        seed=None,  # seeding done in the current function
    )
    attn_mask_valid_indices = batched_attn_mask_indices(
        sparse_tensor, batched_index_tensor
    )

    stacked_index_tensor, query_batch_offsets_2 = padded_to_concatenated(
        batched_index_tensor, query_padding_mask
    )
    assert torch.equal(query_batch_offsets, query_batch_offsets_2)  # sanity check

    linear_index_tensor, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, stacked_index_tensor
    )
    sparse_tensor_values = sparse_tensor.values()

    # Ensure embed_dim is divisible by n_heads
    assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
    head_dim = embed_dim // n_heads

    # Ensure head_dim is even for RoPE
    if use_rope != "none":
        assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

    # Create query tensor: in batched and padded format
    query_tensor = torch.randn(
        len(n_queries), max(n_queries), embed_dim, device=device, dtype=dtype
    )
    stacked_query_tensor, query_batch_offsets_3 = padded_to_concatenated(
        query_tensor, query_padding_mask
    )
    assert torch.equal(query_batch_offsets, query_batch_offsets_3)  # sanity check

    # Create key and value weights
    key_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype)
    value_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype)

    # Create key and value biases if needed
    key_bias: Optional[torch.Tensor] = (
        torch.randn(embed_dim, device=device, dtype=dtype) if use_biases else None
    )
    value_bias: Optional[torch.Tensor] = (
        torch.randn(embed_dim, device=device, dtype=dtype) if use_biases else None
    )

    # Handle RoPE encodings
    batched_key_rope_encoding: Optional[torch.Tensor] = None
    batched_key_positions: Optional[torch.Tensor] = None
    rope_freqs: Optional[torch.Tensor] = None

    if use_rope == "precomputed":
        # Precomputed RoPE encoding: Need to generate for every key for batched
        # then extract the targeted keys for stacked
        # shape: # batch, height, width, level, n_heads, head_dim//2
        batched_key_rope_encoding = torch.randn(
            sparse_tensor.shape[:4] + (n_heads, head_dim // 2),
            device=device,
            dtype=dtype,
        )

    elif use_rope == "from_freqs":
        # On-the-fly RoPE encoding with key positions and frequencies

        # shape: (batch, height, width, level, position_dim)
        batched_key_positions = torch.randn(
            sparse_tensor.shape[:4] + (position_dim,),
            device=device,
            dtype=dtype,
        )

        rope_freqs = torch.rand(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            dtype=dtype,
        )
        # Scale with random magnitude (1-1000) for variety
        rope_freqs *= torch.randint(1, 1000, (1,), device=device)

        # Simple case for freq_groups: one group per position dim
        freq_mask = torch.zeros(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            dtype=torch.bool,
        )
        for pos_dim in range(position_dim):
            freq_mask[pos_dim, pos_dim % n_freq_groups] = True

        rope_freqs *= freq_mask

    # Selection fill if applicable
    if use_selection_fill:
        selection_fill = torch.randn(
            sum(n_queries), 1, embed_dim, device=device, dtype=dtype
        )
    else:
        selection_fill = None

    # Query padding mask if used
    if query_mask_rate is not None and query_mask_rate > 0.0:
        query_mask = (
            torch.rand(stacked_query_tensor.shape[:-1], device=device)
            <= query_mask_rate
        )
    else:
        query_mask = None

    # Random scale factor or None (defaults to 1/sqrt(embed_dim) in the function)
    scale_factor: Optional[float] = (
        torch.rand(1).item() if np.random.random() > 0.5 else None
    )

    # reset seed
    if seed is not None:
        if device.type == "cuda":
            torch.cuda.set_rng_state(torch_rng_state, device)
        else:
            torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

    return {
        "query_tensor": stacked_query_tensor,
        "sparse_tensor": sparse_tensor,
        "index_tensor": stacked_index_tensor,
        "batched_index_tensor": batched_index_tensor,
        "attn_mask_valid_indices": attn_mask_valid_indices,
        "query_padding_mask": query_padding_mask,
        "query_batch_offsets": query_batch_offsets,
        "n_heads": n_heads,
        "sparse_tensor_values": sparse_tensor_values,
        "linear_index_tensor": linear_index_tensor,
        "is_specified_mask": is_specified_mask,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "key_bias": key_bias,
        "value_bias": value_bias,
        "query_mask": query_mask,
        "key_rope_encoding": batched_key_rope_encoding,
        "key_positions": batched_key_positions,
        "rope_freqs": rope_freqs,
        "selection_fill": selection_fill,
        "scale_factor": scale_factor,
        "dropout_p": dropout_p,
        "training": training,
        "metadata": {
            "n_queries": n_queries,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "n_keys_per_query": n_keys_per_query,
            "use_biases": use_biases,
            "use_rope": use_rope,
            "position_dim": position_dim,
            "n_freq_groups": n_freq_groups,
            "unspecified_query_indices": unspecified_query_indices,
            "device": device,
            "dtype": dtype,
        },
    }


def create_sparse_and_index_tensor(
    n_queries: Union[int, list[int]] = 4,
    height: int = 8,
    width: int = 8,
    levels: int = 4,
    embed_dim: int = 16,
    n_keys_per_query: int = 8,
    sparsity: float = 0.9,  # Proportion of empty entries in the sparse tensor
    index_hit_rate: float = 0.75,  # Proportion of indexed values that exist in the sparse tensor
    use_2d_features: bool = False,  # If True, use (n_heads, head_dim) for feature dims
    n_heads: Optional[int] = None,
    unspecified_query_indices: Optional[Union[int, list[int]]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Create a sparse tensor and index tensor for testing batch_sparse_index_subset_attn.

    Args:
        batch_size (int): Number of batches
        height (int): Height of the spatial grid
        width (int): Width of the spatial grid
        levels (int): Number of levels in the hierarchy
        embed_dim (int): Feature dimension
        n_queries (Union[int, list[int]]): Number of queries per batch. Either an int, which
            means a batch_size of 1, or a list of ints with length batch_size.
            For batch_size > 1, the index_tensor will have a pad value of -1 for elements
            index_tensor[i, n_queries_i:]
        n_keys_per_query (int): Number of keys per query
        sparsity (float): Proportion of empty entries in the sparse tensor (0.0-1.0)
        index_hit_rate (float): Proportion of indexed values that exist in the sparse tensor (0.0-1.0)
        use_2d_features (bool): If True, use (n_heads, head_dim) for feature dimensions
        n_heads (int): Number of attention heads (required if use_2d_features=True)
        unspecified_query_indices (Union[int, list[int]], Optional): If given, the
            indicated queries will
        device (Union[str, torch.device]): Device to create tensors on
        dtype (torch.dtype): Data type for the values
        seed (int): Random seed for reproducibility

    Returns:
        tuple[Tensor, Tensor]: The sparse tensor and index tensor
            - sparse_tensor will be of dimension [batch, height, width, levels, embed_dim]
                or [batch, height, width, levels, n_heads, head_dim], with the first 4
                dimensions being sparse dimensions and the last 1 or 2 being dense dims
            - index_tensor will be of dimension
                [batch_size, n_queries_per_batch, n_keys_per_query, 4], with padded and
                designated-unspecified queries having pad values of -1. Note that this
                format has an extra leading batch dimension compared to the stacked-batches
                approach in most of the rest of the code.
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Validate inputs
    if use_2d_features:
        if n_heads is None:
            raise ValueError("n_heads must be provided when use_2d_features=True")
        assert embed_dim % n_heads == 0
        feature_size = (n_heads, embed_dim // n_heads)
    else:
        feature_size = (embed_dim,)

    if isinstance(n_queries, int):
        n_queries = [n_queries]
    batch_size = len(n_queries)
    max_queries = max(n_queries)

    # Calculate the total number of elements in the spatial dimensions
    total_spatial_elements = batch_size * height * width * levels

    # Ensure sparsity is valid
    max_allowed_density = 0.99  # Leave at least 1% of indices as "misses"
    actual_density = min(1.0 - sparsity, max_allowed_density)

    # Decide how many elements will be non-zero based on sparsity
    nnz = int(total_spatial_elements * actual_density)
    nnz = max(1, nnz)  # Ensure at least one element

    # Create indices for all possible positions using meshgrid
    b_range = torch.arange(batch_size, device=device)
    h_range = torch.arange(height, device=device)
    w_range = torch.arange(width, device=device)
    l_range = torch.arange(levels, device=device)

    # Create meshgrid of all possible indices
    b_grid, h_grid, w_grid, l_grid = torch.meshgrid(
        b_range, h_range, w_range, l_range, indexing="ij"
    )

    # Reshape to get a tensor of shape (4, total_elements)
    all_indices = torch.stack(
        [
            b_grid.reshape(-1),
            h_grid.reshape(-1),
            w_grid.reshape(-1),
            l_grid.reshape(-1),
        ],
        dim=0,
    )

    # Randomly select indices for non-zero elements
    perm = torch.randperm(total_spatial_elements, device=device)
    nonsparse_key_indices = all_indices[:, perm[:nnz]]
    sparse_key_indices = all_indices[:, perm[nnz:]]

    # Generate random values for the sparse tensor
    values = torch.randn((nnz,) + feature_size, dtype=dtype, device=device)

    # Create the sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=nonsparse_key_indices,
        values=values,
        size=(batch_size, height, width, levels) + feature_size,
        device=device,
        dtype=dtype,
    ).coalesce()

    index_tensor_batch_shape = (batch_size, max_queries, n_keys_per_query)
    # Now create the index tensor
    # Initialize index tensor with random spatial and level indices
    index_tensor = torch.stack(
        [
            b_range.view(batch_size, 1, 1).expand(-1, max_queries, n_keys_per_query),
            torch.randint(0, height, index_tensor_batch_shape, device=device),
            torch.randint(0, width, index_tensor_batch_shape, device=device),
            torch.randint(0, levels, index_tensor_batch_shape, device=device),
        ],
        dim=-1,
    )
    index_tensor = torch.empty(
        index_tensor_batch_shape + (4,), dtype=torch.long, device=device
    )

    is_hit_mask = torch.rand(index_tensor_batch_shape, device=device) < index_hit_rate

    # For hits, sample from available indices without replacement up to amount of
    # available indices
    for b in range(batch_size):
        nonsparse_this_batch_mask = nonsparse_key_indices[0] == b
        sparse_this_batch_mask = sparse_key_indices[0] == b
        n_available_hits = int(nonsparse_this_batch_mask.sum())
        n_available_misses = int(sparse_this_batch_mask.sum())

        # switch some hits to misses if not enough available indices
        hits_this_batch_mask = is_hit_mask[b]
        hits_this_batch_nz = hits_this_batch_mask.nonzero()
        n_hits_this_batch = min(hits_this_batch_nz.size(0), n_available_hits)
        if n_hits_this_batch < hits_this_batch_nz.size(0):
            # Need to set some hits to misses
            perm = torch.randperm(hits_this_batch_nz.size(0), device=device)
            hits_to_flip = perm[: hits_this_batch_nz.size(0) - n_hits_this_batch]
            hits_this_batch_mask[hits_this_batch_nz[hits_to_flip].unbind(-1)] = False

        n_misses_this_batch = hits_this_batch_mask.numel() - n_hits_this_batch

        # Sample hits
        perm_available_hits = torch.randperm(n_available_hits, device=device)
        sampled_hits_indices = perm_available_hits[:n_hits_this_batch]
        sampled_hits = nonsparse_key_indices[:, nonsparse_this_batch_mask][
            :, sampled_hits_indices
        ]

        # Sample misses
        perm_available_misses = torch.randperm(n_available_misses, device=device)
        sampled_misses_indices = perm_available_misses[:n_misses_this_batch]
        sampled_misses = sparse_key_indices[:, sparse_this_batch_mask][
            :, sampled_misses_indices
        ]

        # Randomly distribute sampled hits and misses among hit/miss key pointers
        # Hits
        hits_nonzero = hits_this_batch_mask.nonzero()
        perm_hit_keys = torch.randperm(hits_nonzero.size(0), device=device)
        hit_pointers = hits_nonzero[perm_hit_keys]

        index_tensor[(b,) + hit_pointers.unbind(-1)] = sampled_hits.T

        # Misses
        misses_nonzero = hits_this_batch_mask.logical_not().nonzero()
        perm_miss_keys = torch.randperm(misses_nonzero.size(0), device=device)
        miss_pointers = misses_nonzero[perm_miss_keys]

        index_tensor[(b,) + miss_pointers.unbind(-1)] = sampled_misses.T

    # Fill in indices past the specified number of queries per batch with -1 pad value
    for i, n_queries_i in enumerate(n_queries):
        index_tensor[i, n_queries_i:] = -1

    # Assert we overwrote the uninitialized data
    assert index_tensor.min() >= -1 and index_tensor.max() <= max(
        batch_size, height, width, levels
    )

    # Fill in the designated unspecified_query_indices (queries with all misses) with -1
    # pad value if requested
    if unspecified_query_indices is not None:
        if isinstance(unspecified_query_indices, int):
            # same single unspecified query per batch
            unspecified_query_indices_nested = [[unspecified_query_indices]] * batch_size
        elif isinstance(unspecified_query_indices[0], int):
            # same one or more unspecified queries per batch
            unspecified_query_indices_nested = [unspecified_query_indices] * batch_size
        assert isinstance(unspecified_query_indices_nested, list)
        assert len(unspecified_query_indices_nested) == len(n_queries) == batch_size
        for b, unspecified_b in enumerate(unspecified_query_indices_nested):
            index_tensor[b, unspecified_b] = -1

    return sparse_tensor, index_tensor


def batched_attn_mask_indices(sparse_tensor: Tensor, index_tensor: Tensor) -> Tensor:
    """Create indices for a boolean attention mask tensor for batched attention
    based on index_tensor

    Args:
        sparse_tensor (Tensor): Sparse tensor output from create_sparse_and_index_tensors
        index_tensor (Tensor): Index tensor output from create_sparse_and_index_tensors

    Returns:
        Tensor: Index tensor of shape [N x 5], where each row is an index of dimension
            [batch, query, i, j, level] corresponding to positions where a query
            attends to a spatial key.
    """
    batch_size, max_queries, _, _ = index_tensor.shape
    _, height, width, levels = sparse_tensor.shape[: sparse_tensor.sparse_dim()]

    nonpad_index_mask = index_tensor[..., 0] != -1  # (batch, query, key)

    batch_indices, query_indices, key_indices = nonpad_index_mask.nonzero(as_tuple=True)

    h, w, lev = index_tensor[batch_indices, query_indices, key_indices, 1:].unbind(-1)

    attn_mask_indices = torch.stack([batch_indices, query_indices, h, w, lev], dim=1)

    return attn_mask_indices

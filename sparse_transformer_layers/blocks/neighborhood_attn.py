from typing import Optional, Union

import torch
from pytorch_sparse_utils.batching import (
    batch_offsets_to_indices,
    seq_lengths_to_batch_offsets,
)
from sparse_transformer_layers.layers.subset_attn import (
    BatchSparseIndexSubsetAttention,
)
from pytorch_sparse_utils.validation import validate_nd
from torch import Tensor, nn

from nd_rotary_encodings import (
    FreqGroupPattern,
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
    prep_multilevel_positions,
)


class SparseNeighborhoodAttentionBlock(nn.Module):
    """Sparse neighborhood attention block for multi-level feature maps.

    This module performs sparse attention over local neighborhoods at
    multiple resolution levels. Each query attends only to keys in its spatial
    neighborhood, with configurable neighborhood sizes at different resolution levels.
    This enables hierarchical feature aggregation while maintaining computational
    efficiency through sparse attention.

    The implementation uses rotary position encodings (RoPE) to incorporate spatial
    and level position information into the attention mechanism. This allows the
    attention to be aware of relative spatial relationships between queries and keys.

    Args:
        embed_dim (int): Dimensionality of input and output embeddings.
        n_heads (int): Number of attention heads.
        n_levels (int, optional): Number of resolution levels. Default: 4
        neighborhood_sizes (Union[Tensor, list[int]], optional): List of odd integers
            specifying the neighborhood size (window width) at each level.
            Default: [3, 5, 7, 9]
        position_dim (int, optional): Dimensionality of spatial positions.
            Default: 2 (for 2D positions).
        dropout (float, optional): Dropout probability for attention weights and
            output projection. Default: 0.0.
        bias (bool, optional): Whether to use bias in linear projections. Default: False.
        norm_first (bool, optional): Whether to apply layer normalization before
            attention. Default: True.
        rope_spatial_base_theta (float, optional): Base theta value for RoPE spatial
            dimensions. Larger values result in lower-frequency rotations, suitable for
            dimensions with greater spatial scale. Default: 100.0.
        rope_level_base_theta (float, optional): Base theta value for RoPE level
            dimension. Default: 10.0.
        rope_share_heads (bool, optional): Whether to share RoPE frequencies across
            attention heads. Default: False.
        rope_freq_group_pattern (str, optional): Pattern to use for grouping RoPE
            frequencies. Options: "single", "partition", "closure". Default: "single".
        rope_enforce_freq_groups_equal (bool, optional): Whether to enforce equal
            division of frequency dimensions across frequency groups. Default: True.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_levels: int = 4,
        neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
        position_dim: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_spatial_base_theta: float = 100.0,
        rope_level_base_theta: float = 10.0,
        rope_share_heads: bool = False,
        rope_freq_group_pattern: Union[str, FreqGroupPattern] = "single",
        rope_enforce_freq_groups_equal: bool = True,
    ):
        super().__init__()
        if len(neighborhood_sizes) != n_levels:
            raise ValueError(
                "Expected len(neighborhood_sizes) to be equal to n_levels, but got "
                f"{len(neighborhood_sizes)=} and {n_levels=}"
            )
        if any(size % 2 != 1 for size in neighborhood_sizes):
            raise ValueError(
                "Expected neighborhood_sizes to be all odd integers, but got "
                f"{neighborhood_sizes}"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.position_dim = position_dim
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(embed_dim)

        self.q_in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.pos_encoding = RoPEEncodingND(
            position_dim + 1,  # +1 for level dimension
            embed_dim,
            n_heads,
            rope_share_heads,
            get_multilevel_freq_group_pattern(position_dim, rope_freq_group_pattern),
            enforce_freq_groups_equal=rope_enforce_freq_groups_equal,
            rope_base_theta=[
                [rope_spatial_base_theta] * position_dim + [rope_level_base_theta]
            ],
        )
        self.subset_attn = BatchSparseIndexSubsetAttention(embed_dim, use_bias=bias)
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

        self.register_buffer(
            "rope_spatial_base_theta", torch.tensor(rope_spatial_base_theta)
        )
        self.register_buffer(
            "rope_level_base_theta", torch.tensor(rope_level_base_theta)
        )
        self.neighborhood_sizes = torch.tensor(neighborhood_sizes, dtype=torch.int)
        self.rope_share_heads = rope_share_heads

    def forward(
        self,
        query: Tensor,
        query_spatial_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        background_embedding: Optional[Tensor] = None,
        query_level_indices: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of sparse neighborhood attention.

        For each query, computes multi-head attention over keys in its spatial
        neighborhood at multiple resolution levels. The neighborhood size at each
        level is determined by the corresponding value in `neighborhood_sizes`.

        Args:
            query (Tensor): Query features of shape [n_queries, embed_dim].
            query_spatial_positions (Tensor): Spatial positions of queries,
                shape [n_queries, position_dim]. The positions must be floating-point
                values scaled in the range of the highest-resolution of the spatial
                shapes. This function will error if the positions are integers. If you
                have integer positions (i.e, position indices), use
                prep_multilevel_positions to get full-resolution decimal positions.
            query_batch_offsets (Tensor): Tensor of shape [batch_size+1]
                indicating where each batch starts in the queries.
            stacked_feature_maps (Tensor): Sparse tensor containing feature maps
                stacked across all levels, with total shape
                [batch, *spatial_dims, levels, embed_dim], where the last dimension
                is dense and the others are sparse.
            level_spatial_shapes (Tensor): Spatial dimensions of each level,
                shape [n_levels, position_dim]. Contains the height, width, etc.
                of feature maps at each resolution level.
            background_embedding (Optional[Tensor]): Optional tensor of shape
                [batch_size, n_levels, embed_dim] to serve as a background embedding.
                If given, then neighborhood indices that are not specified in
                stacked_feature_maps will be given the corresponding background
                embedding for that batch and level. If not given, then these keys will
                be masked out from the queries.
            query_level_indices (Optional[Tensor]): Level indices of each query, shape
                [n_queries]. If None, it defaults to every query being from the level
                of the maximum spatial shape. This value should be specified in the
                encoder, where queries are tokens at various levels, but may be
                unspecified in the decoder, where queries are the object queries that
                are given as being at the full-scale level.
            query_mask (Optional[Tensor]): Optional boolean tensor of shape [n_queries]
                that indicates queries that should not participate in the operation.
                Specifically, if present, positions where this tensor is True will have
                the corresponding query masked out from all keys in the attention
                operation, meaning the query vectors will be unmodified by the
                attention+residual operation.

        Returns:
            Tensor: Output embeddings after neighborhood attention,
                shape [n_queries, embed_dim].

        Raises:
            ValueError: If input tensors don't have expected shapes, or if
                query_spatial_positions is an integer tensor.
        """

        validate_nd(query, 2, "query")
        validate_nd(query_spatial_positions, 2, "query_spatial_positions")
        n_queries = query.shape[0]
        if not torch.is_floating_point(query_spatial_positions):
            raise ValueError(
                "Expected query_spatial_positions to be floating-point, got dtype "
                f"{query_spatial_positions.dtype}"
            )
        if query_spatial_positions.size(1) != self.position_dim:
            raise ValueError(
                "Expected second dim of query_spatial_positions to be equal to "
                f"position dim (={self.position_dim}), but got shape "
                f"{query_spatial_positions.shape}"
            )

        req_feat_map_dim = self.position_dim + 3
        if stacked_feature_maps.ndim != req_feat_map_dim:
            raise ValueError(
                f"Expected stacked_feature_maps to have {req_feat_map_dim} total "
                f"dimensions (batch + position_dims (={self.position_dim}) + level + "
                f"embed_dim), but got shape {stacked_feature_maps.shape}"
            )

        residual = query
        if self.norm_first:
            query = self.norm(query)
        q = self.q_in_proj(query)

        # Prep query: add level dimension to query position for pos encoding.
        query_spatial_level_positions = query_spatial_positions.new_empty(
            (n_queries, self.pos_encoding.position_dim)
        )
        query_spatial_level_positions[:, :-1] = query_spatial_positions
        if query_level_indices is not None:
            query_spatial_level_positions[:, -1] = query_level_indices
        else:
            # We treat each query as living in the full-scale level
            max_spatial_level = level_spatial_shapes.prod(1).argmax(0)
            query_spatial_level_positions[:, -1] = max_spatial_level

        # Position encode queries
        # (key embeddings are rotated inside the custom autograd op)
        query_rotated = self.pos_encoding(q, query_spatial_level_positions)

        # Prepare key data:
        # Compute the neighborhood indices of each query
        nhood_spatial_indices, out_of_bounds_nhood_mask, nhood_level_indices = (
            get_multilevel_neighborhoods(
                query_spatial_positions, level_spatial_shapes, self.neighborhood_sizes
            )
        )
        keys_per_query = nhood_spatial_indices.size(1)
        assert nhood_spatial_indices.shape == (
            n_queries,
            keys_per_query,
            self.position_dim,
        )

        # Initialize the full sparse indices tensor for keys:
        # (batch, *spatial_dims, level)
        key_index_tensor = query_spatial_positions.new_empty(
            n_queries, keys_per_query, self.position_dim + 2, dtype=torch.long
        )
        # get batch indices
        key_batch_indices = batch_offsets_to_indices(query_batch_offsets)

        # index into background_embedding ([bsz, n_levels, embed_dim]) to expanded
        # tensor of same shape as key_index_tensor
        if background_embedding is not None:
            background_embedding = background_embedding[
                key_batch_indices[:, None], nhood_level_indices
            ]

        # expand batch and level indices to broadcasted shape
        key_batch_indices = key_batch_indices[:, None].expand(-1, keys_per_query)
        key_level_indices = nhood_level_indices.unsqueeze(0).expand(n_queries, -1)

        key_index_tensor[:, :, 0] = key_batch_indices
        key_index_tensor[:, :, 1:-1] = nhood_spatial_indices
        key_index_tensor[:, :, -1] = key_level_indices

        # Get the key RoPE components
        key_positions = prep_multilevel_positions(
            nhood_spatial_indices,
            key_batch_indices,
            key_level_indices,
            level_spatial_shapes,
        )
        key_rope_freqs = self.pos_encoding.grouped_rope_freqs_tensor(
            self.pos_encoding.freqs
        )

        x, is_specified_mask = self.subset_attn(
            stacked_feature_maps,
            key_index_tensor,
            query_rotated,
            self.n_heads,
            key_positions=key_positions,
            rope_freqs=key_rope_freqs,
            query_mask=query_mask,
            background_embedding=background_embedding,
        )

        # sanity check that the out of bounds indices were correctly identified as
        # unspecified
        assert torch.all(
            (out_of_bounds_nhood_mask & (~is_specified_mask))
            == out_of_bounds_nhood_mask
        )

        # out projection and residual
        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.q_in_proj.reset_parameters()
        self.subset_attn.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()


def get_multilevel_neighborhoods(
    query_fullscale_spatial_positions: Tensor,
    level_spatial_shapes: Tensor,
    neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes multi-resolution neighborhood indices for query positions.

    Generates neighborhood indices at multiple resolution levels for each query
    position, with configurable neighborhood sizes for each level. This enables
    hierarchical feature aggregation by defining sampling regions around each query
    point at different scales.

    Args:
        query_fullscale_spatial_positions (Tensor): Query positions of shape
            [n_queries, position_dim], where each row contains the N-D position of a
            query point at the full scale resolution.
        level_spatial_shapes (Tensor): Tensor of shape [num_levels, position_dim]
            specifying the spatial dimensions of each resolution level.
        neighborhood_sizes (Union[Tensor, list[int]]): List or tensor of odd integers
            specifying the neighborhood size (window width) at each level.
            Default: [3, 5, 7, 9].

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - multilevel_neighborhood_indices: Tensor of shape
                [n_queries, sum(neighborhood_sizes^position_dim), position_dim]
                containing the spatial indices of all neighborhood points for each
                query across all levels.
            - out_of_bounds_mask: Boolean tensor of shape
                [n_queries, sum(neighborhood_sizes^position_dim)] that is True at locations
                in multilevel_neighborhood_indices that are out of bounds; i.e.
                negative or >= the spatial shape for that level
                If some of the computed neighborhood indices for a query are out of
                bounds of the level's spatial shape, those indices will instead be
                filled with mask values of -1.
            - level_indices: Tensor of shape [sum(neighborhood_sizes^position_dim)]
                mapping each neighborhood position to its corresponding resolution
                level.

    Raises:
        ValueError: If input tensors don't have the expected shape or dimensions, or
            if any neighborhood size is not an odd number.
    """
    validate_nd(
        query_fullscale_spatial_positions, 2, "query_fullscale_spatial_positions"
    )
    n_queries, position_dim = query_fullscale_spatial_positions.shape

    assert (
        level_spatial_shapes.ndim == 2
    ), "This function only supports non-batch-dependent level_spatial_shapes"

    device = query_fullscale_spatial_positions.device

    neighborhood_sizes = torch.as_tensor(neighborhood_sizes, device=device)
    if any(neighborhood_sizes % 2 != 1):
        raise ValueError(
            f"Expected all odd neighborhood_sizes, got {neighborhood_sizes}"
        )

    spatial_scalings = level_spatial_shapes / level_spatial_shapes.max(-2)[0]

    # query x level x position_dim
    query_multilevel_spatial_positions = (
        query_fullscale_spatial_positions.unsqueeze(1) * spatial_scalings
    )

    # Compute neighborhood cardinality for each level
    n_neighborhood_elements = neighborhood_sizes.pow(position_dim)
    total_neighborhood_size = int(n_neighborhood_elements.sum().item())

    # Create the centered neighborhood offset grids for each level
    # [size^position_dim x position_dim] * n_level
    neighborhood_offset_grids = []
    for size in neighborhood_sizes:
        axes = [torch.arange(int(size), device=device)] * position_dim
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        offsets = grid.flatten(0, -2) - (size - 1) / 2
        neighborhood_offset_grids.append(offsets)

    # Prepare level indexing
    level_indices = torch.repeat_interleave(
        torch.arange(neighborhood_sizes.size(0), device=device), n_neighborhood_elements
    )
    level_offsets = seq_lengths_to_batch_offsets(n_neighborhood_elements)

    # Initialize output tensor holding all neighborhood indices
    multilevel_neighborhood_indices = torch.zeros(
        n_queries,
        total_neighborhood_size,
        query_fullscale_spatial_positions.size(-1),
        device=device,
        dtype=torch.long,
    )

    out_of_bounds_mask = torch.zeros(
        n_queries, total_neighborhood_size, device=device, dtype=torch.bool
    )

    # Compute the neighborhood indices and fill in the output tensor
    for level, level_positions in enumerate(
        query_multilevel_spatial_positions.unbind(1)
    ):
        level_start = level_offsets[level]
        level_end = level_offsets[level + 1]
        nhood_grid = neighborhood_offset_grids[level]

        level_neighborhood_indices = level_positions.unsqueeze(
            1
        ).floor().long() + nhood_grid.unsqueeze(0)

        # mask out of bounds indices
        out_of_bounds_mask[:, level_start:level_end] = (
            level_neighborhood_indices < 0
        ).any(-1)
        out_of_bounds_mask[:, level_start:level_end].logical_or_(
            (level_neighborhood_indices >= level_spatial_shapes[level]).any(-1)
        )

        multilevel_neighborhood_indices[:, level_start:level_end, :] = (
            level_neighborhood_indices
        )

    return multilevel_neighborhood_indices, out_of_bounds_mask, level_indices

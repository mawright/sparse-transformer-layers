from typing import Optional

from sparse_transformer_layers.layers.sparse_ms_deform_attn import (
    SparseMSDeformableAttention,
)
from torch import Tensor, nn


class SparseMSDeformableAttentionBlock(nn.Module):
    """A standard transformer block using Sparse Multi-Scale Deformable Attention.

    This module encapsulates the `SparseMSDeformableAttention` layer within a
    typical transformer block structure. It includes a query input projection,
    the attention mechanism itself, an output projection with dropout, a residual
    connection, and layer normalization. The layer normalization can be applied
    either before (pre-norm) or after (post-norm) the main block operations.

    This block is designed to be a plug-and-play component in a larger transformer
    architecture that operates on sparse, multi-scale feature maps, such as the
    encoder or decoder of a Deformable DETR-like model.

    The current version of this module only supports spatially-2D data.

    Args:
        embed_dim (int): The embedding dimension for the queries and features.
        n_heads (int): The number of attention heads.
        n_levels (int): The number of feature levels to sample from.
        n_points (int): The number of sampling points per head per level.
        dropout (float): Dropout probability for the output projection. Defaults to 0.0.
        bias (bool): Whether to include bias terms in the input and output
            projection layers. Defaults to False.
        norm_first (bool): If True, applies layer normalization before the attention
            and projection (pre-norm). If False, applies it after the residual
            connection (post-norm). Defaults to True.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(embed_dim)

        self.q_in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.msdeform_attn = SparseMSDeformableAttention(
            embed_dim,
            n_heads,
            n_levels,
            n_points
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(
        self,
        query: Tensor,
        query_spatial_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        background_embedding: Optional[Tensor] = None,
        query_level_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the SparseMSDeformableAttentionBlock.

        Args:
            query (Tensor): Batch-flattened query tensor of shape [n_query, embed_dim].
            query_spatial_positions (Tensor): Spatial positions of queries,
                shape [n_queries, 2]. The positions must be floating-point
                values scaled to the feature level in which each query resides.
            query_batch_offsets (Tensor): Tensor of shape [batch_size+1] indicating
                the start and end indices for each batch item in the flattened `query`.
            stacked_feature_maps (Tensor): A sparse tensor containing feature maps
                from all levels, with shape [batch, height, width, levels, embed_dim].
                The last dimension is dense, others are sparse.
            level_spatial_shapes (Tensor): Spatial dimensions (height, width) of each
                feature level, shape [n_levels, 2].
            background_embedding (Optional[Tensor]): An embedding to use for sampling
                points that fall in unspecified regions of the sparse feature maps.
                Shape [batch, n_levels, embed_dim].
            query_level_indices (Optional[Tensor]): The level index for each query,
                shape [n_queries]. If None, queries are assumed to be at the largest
                feature level.

        Returns:
            Tensor: The output tensor after the attention block, with the same shape
                as the input `query`, [n_query, embed_dim].
        """
        residual = query
        if self.norm_first:
            query = self.norm(query)

        x = self.q_in_proj(query)

        x = self.msdeform_attn(
            x,
            query_spatial_positions,
            query_batch_offsets,
            stacked_feature_maps,
            level_spatial_shapes,
            query_level_indices,
            background_embedding
            )

        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    def reset_parameters(self):
        """Resets the parameters of all submodules."""
        self.norm.reset_parameters()
        self.q_in_proj.reset_parameters()
        self.msdeform_attn.reset_parameters()
        self.out_proj.reset_parameters()

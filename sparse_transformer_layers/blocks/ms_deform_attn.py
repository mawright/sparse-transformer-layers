from typing import Optional

from sparse_transformer_layers.layers.sparse_ms_deform_attn import (
    SparseMSDeformableAttention,
)
from torch import Tensor, nn


class SparseMSDeformableAttentionBlock(nn.Module):
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
        self.norm.reset_parameters()
        self.q_in_proj.reset_parameters()
        self.msdeform_attn.reset_parameters()
        self.out_proj.reset_parameters()

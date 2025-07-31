from typing import Optional

import torch
from torch import nn, Tensor
import math
from torch.nn.init import constant_, xavier_uniform_

from .utils import sparse_split_heads, multilevel_sparse_bilinear_grid_sample
from pytorch_sparse_utils.batching.batch_utils import batch_offsets_to_indices


## Based on https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
## and https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/multi_scale_deform_attn.py
class SparseMSDeformableAttention(nn.Module):
    """An nn.Module for Multi-Scale Deformable Attention on sparse feature maps.

    This module implements the attention mechanism described in "Deformable DETR".
    Instead of attending to all features in a dense feature map, each query learns
    to sample a small, fixed number of points (`n_points`) from multiple feature
    levels (`n_levels`). The locations of these sampling points are predicted as
    offsets from the query's reference position.

    This implementation is adapted to work with `torch.sparse_coo_tensor`s
    as the input feature maps. It uses a custom bilinear interpolation function
    to efficiently sample values from the sparse feature maps at the predicted
    locations.

    The current version of this module only supports spatially-2D data.

    The module contains learnable parameters for:
    - A value projection (`value_proj`) applied to the input feature maps.
    - A linear layer (`sampling_offsets`) to predict the 2D offsets for each
        sampling point.
    - A linear layer (`attention_weights`) to predict the weight of each sampled
        point.
    - A final output projection (`output_proj`).

    Args:
        embed_dim (int): The embedding dimension of the input and output features.
        n_heads (int): The number of attention heads.
        n_levels (int): The number of feature levels to sample from.
        n_points (int): The number of sampling points per head per level.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(
            embed_dim,
            n_points * n_levels * n_heads * 2,
        )
        self.attention_weights = nn.Linear(embed_dim, n_points * n_levels * n_heads)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(
            self.n_heads,
            dtype=self.sampling_offsets.bias.dtype,
            device=self.sampling_offsets.bias.device,
        ) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.attention_weights.weight.data)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        query_spatial_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        query_level_indices: Optional[Tensor] = None,
        background_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function for SparseMSDeformableAttention.

        Args:
            query (Tensor): Batch-flattened query tensor of shape [n_query x embed_dim]
            query_batch_offsets (Tensor): Tensor of shape [batch_size+1] with values
                such that item i is the start of batch i and item i+1 is the end of
                batch i in the query tensor.
            query_spatial_positions (Tensor): Spatial positions of queries,
                shape [n_queries, position_dim]. The positions must be floating-point
                values scaled in the shape of the feature level in which that query
                resides, as specified by query_level_indices.
            stacked_feature_maps (Tensor): Sparse tensor containing feature maps
                stacked across all levels, with total shape
                [batch, height, width, levels, embed_dim], where the last dimension
                is dense and the others are sparse.
            level_spatial_shapes (Tensor): Spatial dimensions of each level,
                shape [n_levels, 2]. Contains the height and width
                of feature maps at each resolution level.
            query_level_indices (Optional[Tensor]): Level indices of each query, shape
                [n_queries]. If None, it defaults to every query being from the level
                of the maximum spatial shape. This value should be specified in the
                encoder, where queries are tokens at various levels, but may be
                unspecified in the decoder, where queries are the object queries that
                are given as being at the full-scale level.
            background_embedding (Optional[Tensor]): Tensor of shape
                (batch, n_levels, embed_dim) that should be used as an interpolant
                for points that are not specified in stacked_feature_maps. If not given,
                a 0 vector will be used instead.

        Returns:
            Tensor: Output embeddings after sparse deformable attention,
                shape [n_queries, embed_dim].
        """
        n_total_queries = query.shape[0]

        # Perform input value projection
        stacked_feature_maps = torch.sparse_coo_tensor(
            stacked_feature_maps.indices(),
            self.value_proj(stacked_feature_maps.values()),
            size=stacked_feature_maps.shape,
            is_coalesced=stacked_feature_maps.is_coalesced(),
        ).coalesce()

        sampling_offsets: Tensor = self.sampling_offsets(query).view(
            n_total_queries, self.n_points, self.n_levels, self.n_heads, 2
        )

        attention_weights: Tensor = self.attention_weights(query).view(
            n_total_queries, self.n_points * self.n_levels, self.n_heads
        )
        attention_weights = attention_weights.softmax(-2)
        attention_weights = attention_weights.view(
            n_total_queries, self.n_points, self.n_levels, self.n_heads
        )

        scaled_spatial_positions = self._scale_to_multilevel(
            query_spatial_positions, level_spatial_shapes, query_level_indices
        )

        # n_queries, n_points, n_levels, n_heads, 2
        sampling_locations = (
            scaled_spatial_positions[:, None, :, None, :] + sampling_offsets
        )
        sampling_locations = sampling_locations.clamp(
            sampling_locations.new_zeros([]), level_spatial_shapes[:, None, :]
        )

        output = self.multi_scale_deformable_attention(
            stacked_feature_maps,
            level_spatial_shapes,
            sampling_locations,
            query_batch_offsets,
            attention_weights,
            background_embedding=background_embedding,
        )

        output = self.output_proj(output)
        return output

    def _scale_to_multilevel(
        self,
        spatial_positions: Tensor,
        level_spatial_shapes: Tensor,
        level_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Scales single-level spatial positions to multiple levels of different spatial shape.

        Args:
            spatial_positions (Tensor): Tensor of shape [n_points, spatial_dim] where
                each point's position is in the local coordinates of its resident level.
            level_spatial_shapes (Tensor): Tensor of shape [n_levels, spatial_dim] that
                contains the spatial extent of each level.
            level_indices (Optional[Tensor], optional): Tensor of shape [n_points] that
                contains the level in which each point in `spatial_positions` resides.
                If None, a default `level_indices` is used that assumes every point is
                in the feature level of maximum size according to `level_spatial_shapes`.

        Returns:
            Tensor: Tensor of shape [n_points, n_levels, spatial_dim] that rescales each
                point to the coordinate system of each feature level.
        """
        if level_indices is None:
            max_spatial_level = level_spatial_shapes.prod(1).argmax(0)
            level_indices = max_spatial_level.expand(spatial_positions.size(0))

        broadcasted_level_shapes = level_spatial_shapes[level_indices]
        normalized_positions = spatial_positions / broadcasted_level_shapes
        out = normalized_positions.unsqueeze(1) * level_spatial_shapes
        return out

    def multi_scale_deformable_attention(
        self,
        stacked_value_tensors: Tensor,
        level_spatial_shapes: Tensor,
        sampling_locations: Tensor,
        query_batch_offsets: Tensor,
        attention_weights: Tensor,
        background_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        assert isinstance(stacked_value_tensors, Tensor)
        assert stacked_value_tensors.is_sparse
        assert stacked_value_tensors.ndim == 5  # (batch, height, width, level, feature)

        n_queries, n_points, n_levels, n_heads, _ = sampling_locations.shape
        embed_dim = stacked_value_tensors.shape[-1]
        head_dim = embed_dim // n_heads

        stacked_value_tensors = sparse_split_heads(stacked_value_tensors, n_heads)
        # now (batch, height, width, level, heads, head_dim)

        if background_embedding is not None:
            bsz = background_embedding.size(0)
            background_embedding = background_embedding.reshape(
                bsz, n_levels, n_heads, head_dim
            )

        batch_indices: Tensor = batch_offsets_to_indices(query_batch_offsets)
        batch_indices = batch_indices.unsqueeze(-1).expand(-1, self.n_points)

        sampled_values = multilevel_sparse_bilinear_grid_sample(
            stacked_value_tensors,
            sampling_locations,
            batch_indices,
            level_spatial_shapes,
            background_embedding=background_embedding,
        )
        sampled_values = sampled_values.to(attention_weights)

        assert sampled_values.shape == (
            n_queries,
            n_points,
            n_levels,
            n_heads,
            head_dim,
        )
        assert attention_weights.shape == (n_queries, n_points, n_levels, n_heads)

        # do the attention multiplication
        val = sampled_values.permute(0, 3, 4, 1, 2).reshape(
            n_queries * n_heads, head_dim, n_points * n_levels
        )
        w = attention_weights.permute(0, 3, 1, 2).reshape(
            n_queries * n_heads, n_points * n_levels, 1
        )
        output = torch.bmm(val, w).view(n_queries, embed_dim)

        return output

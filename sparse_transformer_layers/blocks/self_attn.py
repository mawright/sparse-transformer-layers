import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from pytorch_sparse_utils.batching import (
    batch_offsets_to_indices,
    concatenated_to_padded,
    padded_to_concatenated,
)
from pytorch_sparse_utils.validation import validate_nd
from torch import Tensor, nn

from nd_rotary_encodings import (
    FreqGroupPattern,
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
    prep_multilevel_positions,
)


class MultilevelSelfAttentionBlockWithRoPE(nn.Module):
    """Self-attention block for multi-level feature maps with rotary position encoding.

    This module applies self-attention across tokens from multiple resolution levels,
    using Rotary Position Encodings (RoPE) to encode the spatial positions of tokens.

    Args:
        embed_dim (int): Dimensionality of input and output embeddings.
        n_heads (int): Number of attention heads.
        position_dim (int, optional): Dimensionality of spatial positions.
            Default: 2 (for 2D positions).
        dropout (float, optional): Dropout probability for attention weights and
            output projection. Default: 0.0.
        bias (bool, optional): Whether to use bias in linear layers. Default: False.
        norm_first (bool, optional): Whether to apply layer normalization before
            attention. Default: True.
        rope_spatial_base_theta (float, optional): Base theta value for RoPE spatial
            dimensions. Default: 100.0.
        rope_level_base_theta (float, optional): Base theta value for RoPE level
            dimension. Default: 10.0.
        rope_share_heads (bool, optional): Whether to share RoPE frequencies across
            attention heads. Default: False.
        rope_freq_group_pattern (str, optional): Pattern to use for grouping RoPE
            frequencies. Options: "single", "partition", "closure". Default: "single".
        rope_enforce_freq_groups_equal (boolean): Passed to the RoPE encoding module.
            Determines whether it will throw an error if the selected
            rope_freq_group_pattern has a number of frequency groups that does not
            evenly divide the RoPE encoding dimensions. Default: True
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
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
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.position_dim = position_dim
        self.norm_first = norm_first
        self.rope_share_heads = rope_share_heads

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
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
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

        self.register_buffer(
            "rope_spatial_base_theta", torch.tensor(rope_spatial_base_theta)
        )
        self.register_buffer(
            "rope_level_base_theta", torch.tensor(rope_level_base_theta)
        )

    def forward(
        self,
        x: Tensor,
        spatial_positions: Tensor,
        level_indices: Tensor,
        level_spatial_shapes: Tensor,
        batch_offsets: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of multi-level self-attention with RoPE.

        Args:
            x (Tensor): Input embeddings of shape (stacked_sequence_length, embed_dim).
                Contains embeddings from all batches concatenated together.
            spatial_positions (Tensor): Spatial positions of each token,
                shape (stacked_sequence_length, position_dim).
                These are expected to be in the original coordinate space of their
                respective levels, NOT normalized to [0, 1] range.
            level_indices (Tensor): Level index for each token,
                shape (stacked_sequence_length, ).
            level_spatial_shapes (Tensor): Spatial dimensions of each level,
                shape (num_levels, position_dim). Contains the height and width
                of feature maps at each resolution level.
            batch_offsets (Tensor): Tensor of shape (batch_size+1, )
                indicating where each batch starts in the stacked sequence.
            attn_mask (Optional[Tensor]): Optional attention mask of shape
                (batch, seq_len, seq_len), (batch*n_heads, seq_len, seq_len), or
                (batch, n_heads, seq_len, seq_len), where True indicates the
                corresponding query/key product should be masked out.

        Returns:
            Tensor: Output embeddings with same shape as input x.

        Raises:
            ValueError: If tensor shapes are incompatible or position dimensions don't match.
        """
        validate_nd(x, 2, "x")  # (stacked sequences x d_model)
        validate_nd(spatial_positions, 2, "spatial_positions")
        if (
            x.shape[0] != spatial_positions.shape[0]
            or x.shape[0] != level_indices.shape[0]
        ):
            raise ValueError(
                "Mismatched sequence lengths: x, spatial_positions, and level_indices "
                f"must have same first dimension, got shapes {x.shape[0]}, "
                f"{spatial_positions.shape[0]}, and {level_indices.shape[0]}"
            )
        if spatial_positions.shape[1] != level_spatial_shapes.shape[1]:
            raise ValueError(
                "Mismatched position dimensions: spatial_positions and level_spatial_shapes "
                f"must have same second dimension, got shapes "
                f"{spatial_positions.shape[1]} and {level_spatial_shapes.shape[1]}"
            )
        if spatial_positions.shape[1] != self.position_dim:
            raise ValueError(
                f"Expected position_dim={self.position_dim}, but spatial_positions has "
                f"shape {spatial_positions.shape}"
            )

        residual = x
        if self.norm_first:
            x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        batch_indices: Tensor = batch_offsets_to_indices(batch_offsets, x.shape[0])

        # normalize spatial positions
        prepped_positions = prep_multilevel_positions(
            spatial_positions, batch_indices, level_indices, level_spatial_shapes
        )

        q, k = self.pos_encoding(q, prepped_positions, k)

        q, pad_mask = concatenated_to_padded(q, batch_offsets)
        k, _ = concatenated_to_padded(k, batch_offsets)
        v, _ = concatenated_to_padded(v, batch_offsets)

        bsz, seq_len, _ = q.shape
        head_dim = self.embed_dim // self.n_heads
        q = q.view(bsz, seq_len, self.n_heads, head_dim)
        k = k.view(bsz, seq_len, self.n_heads, head_dim)
        v = v.view(bsz, seq_len, self.n_heads, head_dim)

        # (batch x seq_len x n_heads x head_dim) -> (batch x n_heads x seq_len x head_dim)
        q: Tensor = q.transpose(1, 2).to(v).contiguous()
        k: Tensor = k.transpose(1, 2).to(v).contiguous()
        v: Tensor = v.transpose(1, 2).contiguous()

        x = self._calc_attn(q, k, v, pad_mask, attn_mask)

        # (batch x n_heads x seq_len x head_dim) ->
        # (batch x seq_len x n_heads x head_dim)
        x = x.transpose(1, 2)

        x = x.reshape(bsz, seq_len, self.embed_dim)
        x, batch_offsets_2 = padded_to_concatenated(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)

        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    def _calc_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pad_mask: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculates multi-head attention with mask handling.

        Args:
            q (Tensor): Query tensor of shape (batch, n_heads, seq_len, head_dim).
            k (Tensor): Key tensor of shape (batch, n_heads, seq_len, head_dim).
            v (Tensor): Value tensor of shape (batch, n_heads, seq_len, head_dim).
            pad_mask (Tensor): Boolean padding mask of shape (batch, seq_len), where
                True indicates a padding element that should be ignored in attention
                computation.
            attn_mask (Optional[Tensor]): Optional attention mask of shape
                (batch, seq_len, seq_len), (batch*n_heads, seq_len, seq_len), or
                (batch, n_heads, seq_len, seq_len), where True indicates the
                corresponding query/key product should be masked out.

        Returns:
            Tensor: Output of shape (batch, n_heads, seq_len, head_dim).

        Raises:
            ValueError: If mask shapes are incompatible.
        """
        bsz, n_heads, seq_len, _ = q.shape
        if pad_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"Expected pad_mask of shape ({bsz, seq_len}), got {pad_mask.shape}"
            )
        if attn_mask is not None and attn_mask.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D or 4D attn_mask, got shape {attn_mask.shape}"
            )
        pad_mask = pad_mask.bool()

        # reshape attn_mask to proper shape if present
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            assert attn_mask.size(0) in (bsz, bsz * n_heads)
            if attn_mask.ndim == 3:
                if attn_mask.size(0) == bsz:
                    attn_mask = attn_mask.view(bsz, 1, seq_len, seq_len)
                else:
                    attn_mask = attn_mask.view(bsz, n_heads, seq_len, seq_len)

        # combine masks
        if pad_mask.any():
            if attn_mask is None:
                not_padding = pad_mask.logical_not()
                # need to split up the batches to avoid needing attn_mask
                # since F.scaled_dot_product_attention can cause memory to blow up by
                # instantiating the full, broadcasted float version of attn_mask
                x = torch.zeros_like(q)
                for i, (q_i, k_i, v_i) in enumerate(zip(q, k, v)):
                    mask_i = not_padding[i]
                    x[i, :, mask_i] = F.scaled_dot_product_attention(
                        q_i[:, mask_i].unsqueeze(0),
                        k_i[:, mask_i].unsqueeze(0),
                        v_i[:, mask_i].unsqueeze(0),
                        dropout_p=self.attn_drop_rate if self.training else 0.0,
                    ).squeeze(0)
                return x
            else:
                # need to combine the pad mask with the attn mask
                pad_mask = pad_mask.view(bsz, 1, 1, seq_len)
                attn_mask = torch.logical_or(attn_mask, pad_mask)

        # True means mask out -> True means should participate in attention
        attn_mask = attn_mask.logical_not() if attn_mask is not None else None

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_rate if self.training else 0.0,
        )
        x = torch.nan_to_num(x, 0.0)
        return x

    # not used, for reference
    def _scaled_dot_product_attention(  # pragma: no cover
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        """Reference implementation of dot product attention to check against.

        Args:
            query (Tensor): Query tensor
            key (Tensor): Key tensor
            value (Tensor): Value tensor
            attn_mask (Optional[Tensor], optional): Attention mask where True indicates
                positions to keep. Defaults to None.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Returns:
            Tensor: Attention output
        """
        scale_factor = 1 / math.sqrt(query.size(-1))

        query = query * scale_factor
        attn_weight = torch.matmul(query, key.transpose(-1, -2))
        if attn_mask is not None:
            attn_weight = torch.masked_fill(attn_weight, ~attn_mask, -torch.inf)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
        return attn_weight @ value

    def reset_parameters(self):
        """Resets parameters to default initializations."""
        self.norm.reset_parameters()
        self.qkv.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()

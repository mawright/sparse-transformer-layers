from typing import Optional

import torch
from torch import Tensor, nn

from pytorch_sparse_utils.indexing.utils import get_sparse_index_mapping

from .autograd import GatherAndSubsetAttentionFunction


@torch.jit.script
def batch_sparse_index_subset_attn(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    query_tensor: Tensor,
    n_heads: int,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    query_mask: Optional[Tensor] = None,
    background_embedding: Optional[Tensor] = None,
    key_rope_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    check_all_specified: bool = False,
):
    """Performs batch selection of elements from a torch sparse tensor followed by
    multi-head attention. Each query attends only to its own specified subset of keys
    (representing that query's local neighborhood, for example).

    The implementation uses a custom autograd function to avoid storing large
    intermediate tensors, recalculating them during the backward pass as needed.

    This function supports rotary position encoding (RoPE) for the selected keys. You
    may either pass in the RoPE encoding tensor `key_rope_encoding`, or its components
    `key_positions` and `rope_freqs`, which allows for computation of
    `key_rope_encoding` inside of the autograd function. This latter option is useful
    because the `rope_encoding` tensor may be significantly larger in memory than
    `key_positions` and `rope_freqs` combined (see shape information in the args
    section below for more detail), and letting the custom autograd function handle
    computing the encoding from the positions and frequencies lets us avoid storing
    that tensor as well.

    Notes:
        - Key indices in index_tensor pointing to spatial locations in sparse_tensor
            that do not have specified values will either be masked out in the attention
            calculation similar to masking out padding in standard attention, or, if
            background_embedding is specified, will be given the corresponding
            background embedding for that query.
        - Queries whose keys are all unspecified will get an output vector of all 0
            in the background_embedding=None case.
        - For rotary position encodings, either provide key_rope_encoding OR both
            key_positions and rope_freqs. Providing both options simultaneously is
            not supported.
        - The output tensor has NOT gone through the output projection (W_o)
            that is encapsulated within most implementations of standard
            multi-head attention. The decision to exclude the output projection
            from this op was driven by the motivation to remove any extra
            complexity that would have diminishing memory performance benefits.
            You will need to add this as an extra nn.Linear layer that gets applied
            to this op's output before it gets passed to a transformer FFN block.
            The residual connection and normalization are also not included.
        - If rotary position encoding is used, query_tensor is expected to be pre-
            rotated before giving it to this function.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension [..., M]; where ... are
            S leading sparse dimensions and M is the dense feature dimension.
        index_tensor (Tensor): Long tensor of dimension [N, L, S]; where N is the
            number of queries, L is the number of keys per query, and S is
            the number of sparse dimensions. Negative indices and indices outside
            the spatial dimension of the sparse tensor are not supported and will
            be considered unspecified.
        query_tensor (Tensor): Query features of shape [N, M]; where N matches
            the number of queries from index_tensor, and M is the feature dimension.
        n_heads (int): Number of attention heads to use.
        key_weight (Tensor): Key projection matrix of shape [M, M].
        value_weight (Tensor): Value projection matrix of shape [M, M].
        key_bias (Optional[Tensor]): Optional bias vector for key projection of shape [M].
        value_bias (Optional[Tensor]): Optional bias vector for value projection of shape [M].
        query_mask (Optional[Tensor]): Optional boolean tensor of shape [N] that
            indicates queries that should not participate in the subset attention
            operation. Specifically, queries at positions where query_mask is True
            will have their output masked out to 0.
        background_embedding (Optional[Tensor]): Optional tensor that will be used as a
            fill value for keys that are not specified in the sparse tensor. Must be
            broadcastable to [N, L, M].
        key_rope_encoding (Optional[Tensor]): Optional positional encoding for keys
            of shape [N, L, n_heads, head_dim]. Used for rotary position embedding (RoPE). The n_heads dimension may also be 1, which will broadcast the
            encoding across heads. If key_rope_encoding is specified, head_dim must be
            divisible by 2. Cannot be used together with key_positions and rope_freqs.
        key_positions (Optional[Tensor]): Position information for each key of shape
            [N, L, P], where N and L match the dimensions from index_tensor
            and P is the dimensionality of the position representation. Used together
            with rope_freqs to compute rotary position embedding (RoPE) from frequencies.
            Cannot be used together with key_rope_encoding.
        rope_freqs (Optional[Tensor]): Frequency values for rotary encodings of shape
            [P, G, n_heads, head_dim] or [P, G, 1, head_dim], where P matches the position
            dimension from key_positions, and G is the number of frequency groups.
            Used together with key_positions to compute rotary position embedding (RoPE)
            from frequencies. Cannot be used together with key_rope_encoding.
            The frequency group dimension allows for grouping of position dimensions
            into specific frequency groups. The intention is to allow dimensions with
            potentially different spatial characteristics (e.g., x and y vs time for
            videos) to be grouped separately. If dimension i is not in frequency group j,
            then rope_freqs[i, j] should be 0. This generalization is experimental and
            under active research. For traditional RoPE, G is 1.
        scale_factor (Optional[float]): Optional scaling factor for attention scores.
            If None, will default is 1/sqrt(M).
        check_all_specified (bool): If True, this function will raise a ValueError
            if any of the indices in `index_tensor` are not specified in `sparse_tensor`.
            If False, unspecified indices will be masked out in the attention calculation.
            Incompatible with background_embedding. Defaults to False.

    Returns:
        - Tensor: Output tensor after attention of shape [N, M], where N is the number
            of queries are the and M is the query embedding dimension.
        - Tensor: Boolean mask of shape [N, L], indicating which keys were actually
            specified in the sparse tensor.
    """
    if index_tensor.is_nested:
        raise ValueError("Nested key index tensor not supported")
        # return __gather_nested_index(sparse_tensor, index_tensor, check_all_specified)

    if query_tensor.shape[:-1] != index_tensor.shape[:-2]:
        raise ValueError(
            "Expected the first n-1 dims of query_tensor and the first n-2 dims of "
            "index_tensor to match, got "
            f"{query_tensor.shape} and {index_tensor.shape}"
        )

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()

    linear_index_tensor, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    attended = GatherAndSubsetAttentionFunction.apply(
        query_tensor,
        n_heads,
        sparse_tensor_values,
        linear_index_tensor,
        is_specified_mask,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        query_mask,
        background_embedding,
        key_rope_encoding,
        key_positions,
        rope_freqs,
        scale_factor,
    )

    assert is_specified_mask.shape == index_tensor.shape[:-1]

    return attended, is_specified_mask


class BatchSparseIndexSubsetAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        use_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=use_bias, dtype=dtype)

    def forward(
        self,
        sparse_tensor: Tensor,
        index_tensor: Tensor,
        query_tensor: Tensor,
        n_heads: int,
        query_mask: Optional[Tensor] = None,
        key_rope_encoding: Optional[Tensor] = None,
        key_positions: Optional[Tensor] = None,
        rope_freqs: Optional[Tensor] = None,
        scale_factor: Optional[float] = None,
        background_embedding: Optional[Tensor] = None,
        check_all_specified: bool = False,
    ):
        kv_params = self.kv_params()

        return batch_sparse_index_subset_attn(
            sparse_tensor,
            index_tensor,
            query_tensor,
            n_heads,
            key_weight=kv_params["key_weight"],
            value_weight=kv_params["value_weight"],
            key_bias=kv_params["key_bias"],
            value_bias=kv_params["value_bias"],
            query_mask=query_mask,
            background_embedding=background_embedding,
            key_rope_encoding=key_rope_encoding,
            key_positions=key_positions,
            rope_freqs=rope_freqs,
            scale_factor=scale_factor,
            check_all_specified=check_all_specified,
        )

    def kv_params(self):
        key_weight, value_weight = self.kv_proj.weight.chunk(2, dim=0)
        if self.kv_proj.bias is not None:
            key_bias, value_bias = self.kv_proj.bias.chunk(2, dim=0)
        else:
            key_bias, value_bias = None, None

        return {
            "key_weight": key_weight,
            "value_weight": value_weight,
            "key_bias": key_bias,
            "value_bias": value_bias,
        }

    def reset_parameters(self):
        self.kv_proj.reset_parameters()

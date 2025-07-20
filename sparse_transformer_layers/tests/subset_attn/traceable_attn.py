from typing import Optional, Any, Union

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from pytorch_sparse_utils.indexing.utils import gather_mask_and_fill
from pytorch_sparse_utils.misc import sparse_tensor_to_dense_with_mask
from sparse_transformer_layers.subset_attn.autograd_helpers import (
    project_kv,
)
from nd_rotary_encodings.forward_backward_fns import (
    calculate_rope,
    rotate_embeddings,
)
from pytorch_sparse_utils.batching import concatenated_to_padded


def traceable_subset_attention(
    query_tensor: Tensor,
    n_heads: int,
    sparse_tensor_values: Tensor,
    linear_index_tensor: Tensor,
    is_specified_mask: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    query_mask: Optional[Tensor] = None,
    selection_fill: Optional[Tensor] = None,
    key_rope_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    batch_kv_projection: bool = True,
    return_extended_outputs: bool = False,
) -> Union[Tensor, dict[str, Union[Tensor, None]]]:
    """Traceable implementation of subset attention using standard Pytorch ops.

    This implementation avoids the memory optimizations of the custom op by using
    straightforward operations that are fully traceable by Pytorch autograd.
    """
    n_queries = query_tensor.size(0)
    embed_dim = query_tensor.size(1)
    n_keys_per_query = linear_index_tensor.size(1)
    head_dim = embed_dim // n_heads

    if scale_factor is None:
        scale_factor = head_dim ** (-1 / 2)

    # Gather values using the same helper as the custom op
    selected = gather_mask_and_fill(
        sparse_tensor_values,
        linear_index_tensor,
        is_specified_mask,
        fill=selection_fill,
    )

    if batch_kv_projection:
        keys, values = project_kv(
            selected, key_weight, value_weight, key_bias, value_bias
        )
    else:
        # Project keys and values separately (no batching)
        keys = F.linear(selected, key_weight, key_bias)
        values = F.linear(selected, value_weight, value_bias)

    # Split heads
    queries = query_tensor.view(n_queries, n_heads, head_dim)
    keys = keys.view(n_queries, n_keys_per_query, n_heads, head_dim)
    values = values.view(n_queries, n_keys_per_query, n_heads, head_dim)

    if key_positions is not None:
        assert rope_freqs is not None
        assert key_rope_encoding is None
        key_rope_encoding = calculate_rope(key_positions, rope_freqs)

    # Apply rotary position encoding if provided
    if key_rope_encoding is not None:
        keys = rotate_embeddings(keys, key_rope_encoding, needs_autograd=True)

    # (n_queries, n_keys_per_query, n_heads, head_dim) ->
    # (n_queries, n_heads, n_keys_per_query, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    values = values.transpose(1, 2).contiguous()

    # Calculate attention scores
    attn_scores = (
        torch.matmul(
            queries.unsqueeze(-2),  # (n_queries, n_heads, 1, head_dim)
            keys.transpose(-1, -2),  # (n_queries, n_heads, head_dim, n_keys_per_query)
        ).squeeze(-2)
        * scale_factor
    )
    # attn_scores: (n_queries, n_heads, n_keys_per_query)

    # Apply masking and softmax
    if query_mask is not None:
        assert query_mask.shape == (n_queries,)
        attn_scores_masked = attn_scores.masked_fill(
            query_mask[:, None, None], -torch.inf
        )
    else:
        attn_scores_masked = attn_scores
    if selection_fill is None:
        assert is_specified_mask.shape == (n_queries, n_keys_per_query)
        attn_scores_masked = attn_scores_masked.masked_fill(
            ~is_specified_mask.unsqueeze(1), -torch.inf
        )
    attn_weights = attn_scores_masked.softmax(-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    attn_weights = F.dropout(attn_weights, dropout_p, training)

    # Apply attention weights to values
    attn_output = torch.matmul(
        attn_weights.unsqueeze(-2),  # (n_queries, n_heads, 1, n_keys_per_query)
        values,  # (n_queries, n_heads, n_keys_per_query, head_dim)
    ).squeeze(-2)
    # output: (n_queries, n_heads, head_dim)

    # Reshape output
    attn_output = attn_output.reshape(n_queries, embed_dim)

    if not return_extended_outputs:
        return attn_output
    else:
        return {
            "queries": queries.flatten(-2, -1),  # n_queries, embed_dim
            # key/value: n_queries, n_keys_per_query, embed_dim
            "keys": keys.transpose(1, 2).flatten(-2, -1),
            "values": values.transpose(1, 2).flatten(-2, -1),
            "is_specified_mask": is_specified_mask,
            "attn_scores": attn_scores,
            "attn_scores_masked": attn_scores_masked,
            "attn_weights": attn_weights,
            "attn_output": attn_output,
            "key_positions": key_positions,
            "key_rope_encoding": key_rope_encoding,
            "rope_freqs": rope_freqs,
        }


def traceable_batched_attention(
    query_tensor: Tensor,
    n_heads: int,
    source_tensor: Tensor,
    attn_mask: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Tensor,
    value_bias: Tensor,
    query_mask: Optional[Tensor] = None,
    selection_fill: Optional[Tensor] = None,
    key_rope_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    return_extended_outputs: bool = False,
) -> Union[Tensor, dict[str, Union[Tensor, None]]]:
    batch_size, n_queries, embed_dim = query_tensor.size()
    n_keys = source_tensor.size(1)
    head_dim = embed_dim // n_heads

    keys, values = project_kv(
        source_tensor, key_weight, value_weight, key_bias, value_bias
    )

    # (batch_size, seq_len, n_heads, head_dim)
    queries = query_tensor.reshape(batch_size, n_queries, n_heads, head_dim)
    keys = keys.reshape(batch_size, n_keys, n_heads, head_dim)
    values: Tensor = values.reshape(batch_size, n_keys, n_heads, head_dim)

    # Compute Rope if needed
    if key_positions is not None:
        assert rope_freqs is not None
        assert key_rope_encoding is None
        key_rope_encoding = calculate_rope(key_positions, rope_freqs)

    # Apply Rope if provided
    if key_rope_encoding is not None:
        key_rope_encoding = key_rope_encoding.view(
            batch_size, n_keys, n_heads, head_dim // 2
        )
        keys: Tensor = rotate_embeddings(keys, key_rope_encoding, needs_autograd=True)

    # (batch_size, n_heads, seq_len, head_dim)
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    if query_mask is not None:
        attn_mask[query_mask] = True

    # (batch_size, n_queries, n_keys) -> (batch_size, 1, n_queries, n_keys)
    attn_mask = attn_mask.unsqueeze(1)

    # attn_output = F.scaled_dot_product_attention(
    attn_output, attn_weights, attn_scores_masked, attn_scores = (
        scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=~attn_mask,
            dropout_p=dropout_p if training else 0.0,
            scale=scale_factor,
        )
    )

    # (batch_size, n_queries, n_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # (batch_size, n_queries, embed_dim)
    attn_output = attn_output.view(batch_size, n_queries, embed_dim)
    if not return_extended_outputs:
        return attn_output
    else:
        return {
            # batch, seq_len, embed_dim
            "queries": queries.transpose(1, 2).flatten(-2, -1),
            "keys": keys.transpose(1, 2).flatten(-2, -1),
            "values": values.transpose(1, 2).flatten(-2, -1),
            "attn_mask": attn_mask,
            "attn_scores": attn_scores,  # batch, n_heads, n_queries, n_keys
            "attn_scores_masked": attn_scores_masked,
            "attn_weights": attn_weights,
            "attn_output": attn_output,
            "key_positions": key_positions,
            "key_rope_encoding": key_rope_encoding,
            "rope_freqs": rope_freqs,
        }


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.expand_as(attn_mask)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_scores = query @ key.transpose(-2, -1) * scale_factor
    attn_scores_masked = attn_scores + attn_bias
    all_masked = torch.isinf(attn_scores_masked).all(-1)

    attn_weight = torch.zeros_like(attn_scores_masked)
    attn_weight[~all_masked] = torch.softmax(attn_scores_masked[~all_masked], dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_output = attn_weight @ value
    return attn_output, attn_weight, attn_scores_masked, attn_scores


def prep_batched_attention(inputs: dict[str, Any]):
    """Preps the inputs for traceable_batched_attention using the dict
    from attention_inputs
    """
    stacked_queries = inputs["query_tensor"]
    batch_offsets = inputs["query_batch_offsets"]
    queries = concatenated_to_padded(stacked_queries, batch_offsets)[0]
    bsz, n_queries, embed_dim = queries.shape
    n_heads = inputs["n_heads"]

    sparse_tensor = torch.sparse_coo_tensor(
        inputs["sparse_tensor"].indices(),
        inputs["sparse_tensor_values"],
        size=inputs["sparse_tensor"].shape,
    )

    source_tensor, source_mask = sparse_tensor_to_dense_with_mask(sparse_tensor)
    _, height, width, n_levels = source_mask.shape

    # make attn_mask: True means invalid attention interaction
    attn_mask = source_tensor.new_ones(
        bsz, n_queries, height, width, n_levels, dtype=torch.bool
    )
    attn_mask_valid_indices = inputs["attn_mask_valid_indices"]
    attn_mask[attn_mask_valid_indices.unbind(1)] = False

    # combine masks
    # [batch, height, width, level] -> [batch, query, height, width, level]
    source_mask = source_mask.unsqueeze(1).expand_as(attn_mask)

    # # source mask: True means valid element - flip and or with attn_mask to combine
    combined_mask = torch.logical_or(attn_mask, source_mask.logical_not())
    combined_mask = combined_mask.view(bsz, n_queries, -1)  # [batch, query, h*w*l]
    # attn_mask = attn_mask.view(bsz, n_queries, -1)

    # flatten source tensor
    source_tensor = source_tensor.view(bsz, -1, embed_dim)  # [batch, h*w*l, embed_dim]
    assert combined_mask.size(-1) == source_tensor.size(1)

    key_weight, value_weight = inputs["key_weight"], inputs["value_weight"]
    key_bias, value_bias = inputs["key_bias"], inputs["value_bias"]

    if inputs["query_mask"] is not None:
        query_mask = concatenated_to_padded(
            inputs["query_mask"].unsqueeze(-1), batch_offsets
        )[0].squeeze(-1)
    else:
        query_mask = None

    key_rope_encoding = inputs["key_rope_encoding"]
    key_positions = inputs["key_positions"]
    rope_freqs = inputs["rope_freqs"]

    scale_factor = inputs["scale_factor"]
    dropout_p = inputs["dropout_p"]
    training = inputs["training"]

    return {
        "query_tensor": queries,
        "n_heads": n_heads,
        "source_tensor": source_tensor,
        "attn_mask": combined_mask,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "key_bias": key_bias,
        "value_bias": value_bias,
        "query_mask": query_mask,
        "key_rope_encoding": key_rope_encoding,
        "key_positions": key_positions,
        "rope_freqs": rope_freqs,
        "scale_factor": scale_factor,
        "dropout_p": dropout_p,
        "training": training,
    }

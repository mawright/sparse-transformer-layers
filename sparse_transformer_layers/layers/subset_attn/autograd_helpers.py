from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from pytorch_sparse_utils.indexing.utils import gather_mask_and_fill


@torch.jit.script
def split_heads(tensor: Tensor, n_heads: int) -> Tensor:
    """Splits the last dimension of a tensor into multiple attention heads.

    Args:
        tensor (Tensor): Input tensor to split, of shape
            [n_queries, embed_dim] (for query) or
            [n_queries, n_keys_per_query, embed_dim] (for key or value)
        n_heads (int): Number of attention heads

    Returns:
        Tensor: Reshaped tensor with last dimension split to
            [n_queries, n_heads, embed_dim / n_heads] (for query) or
            [n_queries, n_keys_per_query, n_heads, embed_dim / n_heads] (for key or value)
    """
    if tensor.ndim not in (2, 3):
        error_str = "Expected tensor to be 2D or 3D, got shape " + str(tensor.shape)
        raise ValueError(error_str)

    tensor_shape = tensor.size()
    if tensor_shape[-1] % n_heads != 0:
        error_str = "Last dimension of tensor with shape " + str(tensor_shape)
        error_str += "cannot be evenly split into n_heads "
        error_str += "(" + str(n_heads) + ") heads."
        raise ValueError(error_str)

    new_shape = tensor_shape[:-1] + (n_heads, tensor_shape[-1] // n_heads)
    split_tensor = tensor.reshape(new_shape)

    return split_tensor


@torch.jit.script
def permute_for_attention(tensor: Tensor) -> Tensor:
    """Permutes dimensions of tensor for batched-heads attention computation.

    Args:
        tensor (Tensor): Input tensor to permute, of shape
        [n_queries, n_heads, head_dim] (for query) or
        [n_queries, n_keys_per_query, n_heads, head_dim] (for key or value)

    Returns:
        Tensor: Permuted tensor with n_heads dimension moved to first position, of
            shape [n_heads, n_queries, head_dim] (for query) or
            [n_heads, n_queries, n_keys_per_query, head_dim] (for key or value)
    """
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected tensor to be 3D or 4D, got shape {tensor.shape}")
    if tensor.ndim == 3:
        # For query: [n_queries, n_heads, head_dim] -> [n_heads, n_queries, head_dim]
        return tensor.transpose(-2, -3).contiguous()
    else:
        # For key/value: [n_queries, n_keys_per_query, n_heads, head_dim] ->
        # [n_heads, n_queries, n_keys_per_query, head_dim]
        # standard batched-heads approach for multiplication with q and attn_weights
        # but with added n_keys_per_query dim that keys broadcasts over q and values
        # contracts with attn_weights
        return tensor.permute(2, 0, 1, 3).contiguous()


@torch.jit.script
def permute_for_attention_backward(tensor: Tensor) -> Tensor:
    """Permutes dimensions of tensor that was in batched-heads format back to
    standard format.

    Args:
        tensor (Tensor): Input tensor to permute, of shape
        [n_heads, n_queries, head_dim] (for query) or
        [n_heads, n_queries, n_keys_per_query, head_dim] (for key or value)

    Returns:
        Tensor: Permuted tensor with n_heads dimension moved to first position, of
            shape [n_queries, n_heads, head_dim] (for query) or
            [n_queries, n_keys_per_query, n_heads, head_dim] (for key or value)
    """
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected tensor to be 3D or 4D, got shape {tensor.shape}")
    if tensor.ndim == 3:
        # For query: [n_heads, n_queries, head_dim] -> [n_queries, n_heads, head_dim]
        return tensor.transpose(-2, -3).contiguous()
    else:
        # For key/value: n_heads, n_queries, n_keys_per_query, head_dim]
        # -> [n_queries, n_keys_per_query, n_heads, head_dim]
        return tensor.permute(1, 2, 0, 3).contiguous()


@torch.jit.script
def project_kv(
    source_elements: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Efficiently computes the key and value tensors for attention.

    Args:
        source_elements (Tensor): Source elements (pre-kv projection) for attention
            with shape [..., embed_dim], where ... are arbitrary leading batch dims.
        key_weight (Tensor): Key projection matrix of shape [embed_dim, embed_dim]
        value_weight (Tensor): Value projection matrix of shape [embed_dim, embed_dim]
        key_bias (Optional[Tensor]): Key projection bias of shape [embed_dim]
        value_bias (Optional[Tensor]): Value projection bias of shape [embed_dim]

    Returns:
        - keys (Tensor): Key tensor of shape [..., embed_dim]
        - values (Tensor): Value tensor of shape [..., embed_dim]"""
    # Stack weight matrices to batch the keys and values projections
    weights_stacked = torch.cat([key_weight, value_weight])  # (2*embed_dim, embed_dim)

    # Handle stacking of biases if present
    if key_bias is not None or value_bias is not None:
        key_bias = (
            key_bias
            if key_bias is not None
            else key_weight.new_zeros(key_weight.size(0))
        )
        value_bias = (
            value_bias
            if value_bias is not None
            else value_weight.new_zeros(value_weight.size(0))
        )
        biases_stacked = torch.cat([key_bias, value_bias])  # (2*embed_dim)
    else:
        biases_stacked = None

    # (..., 2*embed_dim)
    kv = F.linear(source_elements, weights_stacked, biases_stacked)
    keys, values = kv.chunk(2, dim=-1)  # (..., embed_dim) * 2

    return keys, values


@torch.jit.script
def select_values_and_project_kv(
    sparse_tensor_values: Tensor,
    linear_index_tensor: Tensor,
    is_specified_mask: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor],
    value_bias: Optional[Tensor],
    fill: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Retrieves source sequence elements and computes the key and value tensors for
    multi-head attention.

    Args:
        sparse_tensor_values (Tensor): Values from sparse tensor of shape
            [num_sparse_values, embed_dim]
        linear_index_tensor (Tensor): Long tensor of shape [n_queries, n_keys_per_query]
            with elements corresponding to the indices of each key along
            sparse_tensor_values's first dimension. If created by
            get_sparse_index_mapping, indices of unspecified keys will be
            masked to 0 to potentially speed up lookup.
        is_specified_mask (Tensor): Boolean mask of shape
            [n_queries, n_keys_per_query] indicating which indices are
            specified in the sparse tensor
        key_weight (Tensor): Key projection matrix of shape [embed_dim, embed_dim]
        value_weight (Tensor): Value projection matrix of shape [embed_dim, embed_dim]
        key_bias (Optional[Tensor]): Key projection bias of shape [embed_dim]
        value_bias (Optional[Tensor]): Value projection bias of shape [embed_dim]
        fill (Optional[Tensor]): Optional fill value to be put in the selected
            pre-projection tensor in locations corresponding to linear_index_tensor
            being False. If None, a default fill value of 0 is used. Must be
            broadcastable to the final shape.

    Returns:
        - keys (Tensor): Key tensor of shape
            [n_queries, n_keys_per_query, embed_dim]
        - values (Tensor): Value tensor of shape
            [n_queries, n_keys_per_query, embed_dim]
        - selected (Tensor): Selected features from sparse tensor before keys and values
            projections, of shape [n_queries, n_keys_per_query, embed_dim]
    """
    assert linear_index_tensor.ndim == 2
    assert sparse_tensor_values.ndim == 2

    selected = gather_mask_and_fill(sparse_tensor_values, linear_index_tensor, is_specified_mask, fill=fill)

    keys, values = project_kv(selected, key_weight, value_weight, key_bias, value_bias)
    return keys, values, selected

@torch.jit.script
def linear_grads(
    grad_output: Optional[Tensor],
    inputs: Tensor,
    need_weight_grad: bool,
    need_bias_grad: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Efficiently computes gradients for weights and biases of a linear layer.
    Computes only the gradients required. If both the weight and bias gradient
    are required, computes them efficiently with the bias trick by concatenating
    a column of 1s onto the weight matrix before matmuling. The product of matmuling
    this augmented matrix with the gradient is then the weight and bias gradients
    stacked together.

    This function supports both regular and stacked gradients. When grad_output
    is 3D, with the leading dimension representing a stacking of the keys and values
    gradients, the returned tensors are 3D and 2D, respectively.

    Args:
        grad_output (Optional[Tensor]): Gradient of output, of shape
            [batch_size, out_features], or
            [num_projections, batch_size, out_features] for stacked mode, or None.
            If grad_output is None, this function returns (None, None).
        inputs (Tensor): Input tensor, of shape [batch_size, in_features]
        need_weight_grad (bool): Whether weight gradients are needed
        need_bias_grad (bool): Whether bias gradients are needed

    Returns:
        - weight_grad (Optional[Tensor]): Gradient for weights, of shape
            [out_features, in_features] for non-stacked mode,
            [num_projections, out_features, in_features] for stacked mode,
            or None if need_weight_grad is False
        - bias_grad (Optional[Tensor]): Gradient for bias, of shape
            [out_features] for non-stacked mode, [num_projections, out_features]
            for stacked mode, or None if need_bias_grad is False
    """
    if grad_output is None:
        return None, None

    if not (grad_output.ndim == 2 or grad_output.ndim == 3):
        raise ValueError(
            f"Expected grad_output.ndim to be 2 or 3, got {grad_output.ndim}"
        )

    is_stacked_mode = grad_output.ndim == 3

    if need_weight_grad and need_bias_grad:
        # Set up bias trick
        ones = inputs.new_ones(inputs.size(0), 1)
        augmented_input = torch.cat([inputs, ones], dim=1)

        if is_stacked_mode:
            # fmt: off
                combined_grad = torch.bmm(
                    grad_output.transpose(-1, -2), # (num_proj, out_features, batch_size)
                    augmented_input.unsqueeze(0).expand(
                        grad_output.size(0), -1, -1
                    ),  # (num_proj, batch_size, in_features+1)
                )  # (num_proj, out_features, in_features+1)
        # fmt: on
        else:
            combined_grad = torch.mm(grad_output.t(), augmented_input)
        return combined_grad[..., :-1], combined_grad[..., -1]
    elif need_weight_grad:
        if is_stacked_mode:
            # fmt: off
                weight_grad = torch.bmm(
                    grad_output.transpose(-1, -2),  # (num_proj, out_features, batch_size)
                    inputs.unsqueeze(0).expand(
                        grad_output.size(0), -1, -1
                    ), # (num_proj, batch_size, in_features)
                )  # (num_proj, out_features, in_features)
        # fmt: on
        else:
            weight_grad = torch.mm(grad_output.t(), inputs)
        return weight_grad, None
    elif need_bias_grad:
        bias_grad = grad_output.sum(-2)
        return None, bias_grad
    return None, None

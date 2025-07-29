# pyright: reportAttributeAccessIssue=false
from typing import Optional, Union

import torch
from nd_rotary_encodings.functional import (
    calculate_rope,
    calculate_rope_backward,
    rotate_embeddings,
    rotate_embeddings_backward,
)
from torch import Tensor
from torch.amp import (
    custom_bwd,  # pyright: ignore[reportPrivateImportUsage]
    custom_fwd,  # pyright: ignore[reportPrivateImportUsage]
)

from .autograd_helpers import (
    linear_grads,
    permute_for_attention,
    permute_for_attention_backward,
    select_values_and_project_kv,
    split_heads,
)


class GatherAndSubsetAttentionFunction(torch.autograd.Function):
    """Custom autograd function that implements memory-efficient attention
    where each query attends to its own local subset of keys. This implementation
    uses optimized gradient checkpointing, avoiding keeping large intermediate
    tensors in memory by recalculating them during the backward pass. This saves
    significant memory for only a minor increase in time to run the backward.
    """

    @staticmethod
    def _forward_shared(
        query_tensor: Tensor,
        sparse_tensor_values: Tensor,
        linear_index_tensor: Tensor,
        is_specified_mask: Tensor,
        key_weight: Tensor,
        value_weight: Tensor,
        key_bias: Optional[Tensor],
        value_bias: Optional[Tensor],
        selection_fill: Optional[Tensor],
        n_heads: int,
        key_positions: Optional[Tensor],
        rope_freqs: Optional[Tensor],
        key_rope_encoding: Optional[Tensor],
        is_for_backward: bool,  # return additional values if True
        need_backward_key_branch: Optional[bool] = False,
    ) -> Union[
        tuple[Tensor, Tensor, Tensor, Optional[Tensor]],  # is_for_backward False
        tuple[
            Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]
        ],  # is_for_backward True
    ]:
        """Forward pass computations that get repeated in the backward pass"""

        #### Forward step 3: Sparse values selection and input projection
        # fmt: off
        keys, values, selected = select_values_and_project_kv(
            sparse_tensor_values, linear_index_tensor, is_specified_mask,
            key_weight, value_weight, key_bias, value_bias, selection_fill
        )
        # fmt: on
        if not is_for_backward:
            del selected

        queries: Tensor = split_heads(query_tensor, n_heads)
        keys: Tensor = split_heads(keys, n_heads)
        values: Tensor = split_heads(values, n_heads)

        #### Forward step 4: RoPE encoding calculation

        if key_positions is not None and rope_freqs is not None:
            assert key_rope_encoding is None
            key_rope_encoding = calculate_rope(key_positions, rope_freqs)

        #### Forward step 5: Rotate the keys by applying RoPE

        keys_unrotated_copy = None
        if key_rope_encoding is not None:
            if is_for_backward and need_backward_key_branch:
                # used later in backward pass to compute RoPE grads
                keys_unrotated_copy = keys.clone()
            keys = rotate_embeddings(keys, key_rope_encoding, needs_autograd=False)

        #### Forward step 6: Permutation

        # (n_heads, n_queries, head_dim)
        queries = permute_for_attention(queries)

        # (n_heads, n_queries, n_keys_per_query, head_dim)
        keys = permute_for_attention(keys)
        values = permute_for_attention(values)

        if not is_for_backward:
            return queries, keys, values, key_rope_encoding
        else:
            return (
                queries,
                keys,
                values,
                key_rope_encoding,
                selected,
                keys_unrotated_copy,
            )

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
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
        scale_factor: Optional[float] = None,  # scaling for attn, default 1/sqrt(d)
        dropout_p: float = 0.0,
        training: bool = True,
    ) -> Tensor:
        """Performs sparse neighborhood attention with minimal memory usage.

        This function computes attention where each query attends only to its
        local neighborhood of keys, without materializing the full attention matrix
        or storing intermediate tensors.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context to save tensors for backward
            query_tensor (Tensor): Query features of shape [n_queries, embed_dim]
            n_heads (int): Number of attention heads
            sparse_tensor_values (Tensor): Values from sparse tensor of shape
                [num_sparse_values, embed_dim]
            linear_index_tensor (Tensor): Long tensor of shape
                [n_queries, n_keys_per_query] with elements corresponding to the
                indices of each key/value element along sparse_tensor_values's first
                dimension. If created by get_sparse_index_mapping, indices of
                unspecified keys will be masked to 0 to potentially speed up lookup.
            is_specified_mask (Tensor): Boolean mask of shape
                [n_queries, n_keys_per_query] indicating which indices are
                specified in the sparse tensor
            key_weight (Tensor): Key projection matrix of shape [embed_dim, embed_dim]
            value_weight (Tensor): Value projection matrix of shape [embed_dim, embed_dim]
            key_bias (Optional[Tensor]): Key projection bias of shape [embed_dim]
            value_bias (Optional[Tensor]): Value projection bias of shape [embed_dim]
            query_mask: (Optional[Tensor]): Tensor of shape [n_queries] that
                indicates queries that should not participate in this operation.
                Specifically, if present, positions where this tensor is True will have
                the corresponding query in query_tensor masked out from every key,
                meaning that this operation's output tensor at those positions will be
                equal to the input values. This is potentially useful for query tensors
                where some queries represent "virtual" queries without a well-defined
                neighborhood, such as <cls> tokens or similar.
            selection_fill (Optional[Tensor]): Tensor that will be used as a fill value
                in the pre-projection key/value tensor at locations where is_specified_mask
                is False. Must be broadcastable to [n_queries, n_keys_per_query, embed_dim].
                If None, a fill value of 0 will be used.
            key_rope_encoding (Optional[Tensor]): Positional encoding for keys of shape
                [n_queries, n_keys_per_query, n_heads, head_dim/2]. Used for rotary
                position embedding (RoPE). Cannot be used together with key_positions
                and rope_freqs.
            key_positions (Optional[Tensor]): Position information for each key of
                shape [n_queries, n_keys_per_query, position_dim]. Used together with
                rope_freqs to compute rotary position embedding (RoPE) from frequencies.
                Cannot be used together with key_rope_encoding.
            rope_freqs (Optional[Tensor]): Frequency values for rotary embeddings of
                shape [position_dim, n_freq_groups, n_heads, head_dim/2] or
                [position_dim, n_freq_groups, 1, head_dim/2]. Used together with
                key_positions to compute rotary position embedding (RoPE) from
                frequencies. Cannot be used together with key_rope_encoding.
                This implementation allows for grouping of position dimensions into
                specific frequency groups. The intention is to allow dimensions with
                potentially different spatial characteristics (e.g., x and y vs time
                for videos) to be grouped separately. This generalization is
                experimental and under active research. If dimension i is not in
                frequency group j, then rope_freqs[i, j] should be 0.
                For traditional RoPE, keep n_freq_groups as 1.
            scale_factor (Optional[float]): Scaling factor for attention scores.
                Default is 1/sqrt(head_dim).
            dropout_p (float): Dropout rate for attention weights.
            training (bool): Whether we are in training mode. If True, dropout is
                applied.

        Returns:
            Tensor: Output tensor after attention of shape [n_queries, embed_dim]

        Note:
            - The output tensor has NOT gone through the output projection (W_o)
                that is encapsulated within most implementations of standard
                multi-head attention. The decision to exclude the output projection
                from this op was driven by the motivation to remove any extra
                complexity that would have diminishing memory performance benefits.
                You will need to add this as an extra nn.Linear layer that gets applied
                to this op's output before it gets passed to a transformer FFN block.
                The residual connection and normalization are also not included.
        """

        #########
        # The forward pass has several steps:
        # 1. Shape checks
        # 2. Setting up for the backward pass (saving tensors and shape ints)
        # 3. Input projection: Retrieving the input values from sparse_tensor_values
        #   using index_tensor. Then, as in standard multi-head attention, pass these
        #   values through the key and value projections and unstack the heads of the
        #   resulting tensors
        # 4. If we have the two RoPE inputs (key positions and RoPE freqs), compute
        #   the RoPE encoding rotation vector
        # 5. If we computed the RoPE rotation vector or were given it, apply it to keys
        # 6. Permute the dimensions of the queries, keys, and values tensors
        #   to heads-batched order (similar to standard MHA)
        # 7. Compute the attention scores by matmuling queries and keys along the head_dim
        #   dimension (with scale factor) (again similar to standard MHA)
        # 8. Mask out the unspecified keys so queries can't attend to them and
        #   apply softmax to compute attention weights (the masking is slightly
        #   different from standard MHA due to the per-query subset structure of
        #   the keys but mostly straightforward)
        # 9. Compute the output values by matmuling the attention weights with
        #   the value tensor, contracting out the n_queries_per_keys dimension.

        # The selected, keys, and values tensors are the major memory consumers because they
        # are n_keys_per_query times larger than the queries and attn_scores tensors. The
        # custom autograd op lets us have each query element attend to its own
        # subset of key elements. In tensor terms, this means that the keys and values
        # tensors are 4D while the queries (query) tensor is 3D as in standard multi-head
        # attention. The matmuls between queries and keys/values are performed with unsqueezes and
        # broadcasts.

        ctx.set_materialize_grads(False)

        #### Step 1: shape checks

        assert query_tensor.ndim == 2  # (n_queries, embed_dim)
        assert linear_index_tensor.ndim == 2  # (n_queries, n_keys_per_query)

        n_queries = query_tensor.size(0)
        embed_dim = query_tensor.size(1)
        n_keys_per_query = linear_index_tensor.size(1)
        head_dim = embed_dim // n_heads

        assert query_tensor.size(0) == linear_index_tensor.size(0) == n_queries
        assert linear_index_tensor.shape == is_specified_mask.shape
        assert key_weight.ndim == 2
        assert value_weight.ndim == 2

        # embed_dim
        # kv projection
        assert (
            key_weight.size(1)
            == value_weight.size(1)
            == sparse_tensor_values.size(-1)
            == embed_dim
        )
        # attn calculation
        assert key_weight.size(0) == query_tensor.size(1) == embed_dim

        if query_mask is not None:
            assert query_mask.ndim == 1
            assert query_mask.shape[0] == n_queries
            assert query_mask.dtype == torch.bool

        # rope validations
        if key_rope_encoding is not None and (
            key_positions is not None or rope_freqs is not None
        ):
            raise ValueError(
                "Cannot provide both key_rope_encoding and (key_positions, rope_freqs)"
            )
        if (key_positions is not None) ^ (rope_freqs is not None):
            raise ValueError("Cannot provide only one of key_positions and rope_freqs")

        if key_rope_encoding is not None:
            assert head_dim % 2 == 0, "head_dim must be even to use RoPE"
            assert key_rope_encoding.shape == (
                n_queries,
                n_keys_per_query,
                n_heads,
                head_dim / 2,
            )

        if key_positions is not None and rope_freqs is not None:
            # check shapes
            assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

            # (n_queries, n_keys_per_query, position_dim)
            assert key_positions.ndim == 3

            # (position_dim, n_groups, n_heads or 1, head_dim)
            assert rope_freqs.ndim == 4

            assert key_positions.shape[0] == n_queries
            assert key_positions.shape[1] == n_keys_per_query
            assert rope_freqs.shape[-1] == head_dim / 2

            position_dim = key_positions.shape[-1]
            assert rope_freqs.shape[0] == position_dim

        #### Step 2: backward pass preparation

        # save shape info
        ctx.n_queries = n_queries
        ctx.embed_dim = embed_dim
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim
        ctx.n_keys_per_query = n_keys_per_query

        # save dropout info
        ctx.dropout_p = dropout_p
        ctx.training = training

        # default scale factor
        if scale_factor is None:
            scale_factor = head_dim ** (-1 / 2)
        ctx.scale_factor = scale_factor

        # save tensors
        ctx.save_for_backward(
            query_tensor,
            sparse_tensor_values,
            key_weight,
            value_weight,
            key_bias,  # pyright: ignore[reportArgumentType]
            value_bias,  # pyright: ignore[reportArgumentType]
            query_mask,  # pyright: ignore[reportArgumentType]
            selection_fill,  # pyright: ignore[reportArgumentType]
            (
                key_rope_encoding
                if not (key_positions is not None and rope_freqs is not None)
                else None
            ),  # pyright: ignore[reportArgumentType]
            key_positions,  # pyright: ignore[reportArgumentType]
            rope_freqs,  # pyright: ignore[reportArgumentType]
        )
        ctx.index_tensor = linear_index_tensor
        ctx.is_specified_mask = is_specified_mask

        #### Steps 3-6 get repeated by backward so step into a shared helper function

        (
            queries,
            keys,
            values,
            key_rope_encoding,  # pyright: ignore[reportAssignmentType]
        ) = GatherAndSubsetAttentionFunction._forward_shared(
            query_tensor,
            sparse_tensor_values,
            linear_index_tensor,
            is_specified_mask,
            key_weight,
            value_weight,
            key_bias,
            value_bias,
            selection_fill,
            n_heads,
            key_positions,
            rope_freqs,
            key_rope_encoding,
            is_for_backward=False,
        )

        #### Step 7: Attention scores calculation

        queries = queries.unsqueeze(-2)  # (n_heads, n_queries, 1, head_dim)
        # disable black formatter to keep dimension comments aligned
        # fmt: off
        attn_scores = torch.matmul(
            queries * scale_factor, # (n_heads, n_queries, 1, head_dim)
            keys.transpose(-1, -2)  # (n_heads, n_queries, head_dim, n_keys_per_query)
        ).squeeze(-2)               # (n_heads, n_queries, n_keys_per_query)
        # fmt: on

        #### Step 8: Masking and softmax

        # Apply query padding mask if present
        if query_mask is not None:
            # Mask out rows from the attention scores for padded queries
            # attn_scores: [n_heads, n_queries, n_keys_per_query]
            attn_scores.masked_fill_(query_mask[None, :, None], -torch.inf)

        if selection_fill is None:
            attn_scores.masked_fill_(~is_specified_mask, -torch.inf)
        attn_weights = attn_scores.softmax(-1)
        # nans expected if all of the keys that a query tried to attend to were unspecified
        attn_weights.nan_to_num_(0.0)

        ctx.attn_weights = attn_weights  # save pre-dropout weights

        #### Step 8.5: Apply dropout if in training mode
        attn_dropout_mask = None
        if training and dropout_p > 0.0:
            # Create and apply dropout mask
            attn_dropout_mask = torch.empty_like(attn_weights, dtype=torch.bool)
            attn_dropout_mask.bernoulli_(dropout_p)  # 1 means drop this element
            dropout_scale = 1.0 / (1.0 - dropout_p)

            attn_weights_dropped = attn_weights.clone()
            attn_weights_dropped.masked_fill_(attn_dropout_mask, 0.0)
            attn_weights_dropped *= dropout_scale

            attn_weights = attn_weights_dropped

        ctx.attn_dropout_mask = attn_dropout_mask

        #### Step 9: Compute the output values

        # fmt: off
        output = torch.matmul(
            attn_weights.unsqueeze(-2), # (n_heads, n_queries, 1, n_keys_per_query)
            values,                     # (n_heads, n_queries, n_keys_per_query, head_dim)
        ).squeeze(-2)                   # (n_heads, n_queries, head_dim)
        # fmt: on

        output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
        output = output.reshape(n_queries, embed_dim)

        return output

    @staticmethod
    def _initialize_backward(
        ctx: torch.autograd.function.FunctionCtx,
    ) -> tuple[
        # fmt: off
        #  (return type hint matches lines of return)
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        int, int,
        float, dict[str, bool], float, bool, Optional[Tensor]
        # fmt: on
    ]:
        """Retrieves all saved tensors and shape info from forward pass and preps
        backward variables
        """
        # retrieve tensors
        # fmt: off
        (
            query_tensor, sparse_tensor_values, key_weight, value_weight,
            key_bias, value_bias, query_mask, selection_fill,
            key_rope_encoding, key_positions, rope_freqs,
        ) = ctx.saved_tensors
        # fmt: on
        index_tensor: Tensor = ctx.index_tensor
        is_specified_mask: Tensor = ctx.is_specified_mask

        attn_weights: Tensor = ctx.attn_weights

        # retrieve shape info
        embed_dim: int = ctx.embed_dim
        n_heads: int = ctx.n_heads

        # retrieve scale factor
        scale_factor: float = ctx.scale_factor

        # retrieve dropout info
        dropout_p: float = ctx.dropout_p
        training: bool = ctx.training
        attn_dropout_mask: Optional[Tensor] = ctx.attn_dropout_mask

        # account for which inputs need gradients
        needed_grads = {
            "query": ctx.needs_input_grad[0],
            "sparse_values": ctx.needs_input_grad[2],
            "key_weight": ctx.needs_input_grad[5],
            "value_weight": ctx.needs_input_grad[6],
            "key_bias": key_bias is not None and ctx.needs_input_grad[7],
            "value_bias": value_bias is not None and ctx.needs_input_grad[8],
            "selection_fill": selection_fill is not None and ctx.needs_input_grad[10],
            "key_rope_encoding": (
                key_rope_encoding is not None and ctx.needs_input_grad[11]
            ),
            "key_positions": (key_positions is not None and ctx.needs_input_grad[12]),
            "rope_freqs": rope_freqs is not None and ctx.needs_input_grad[13],
        }

        # fmt: off
        return (
            query_tensor, sparse_tensor_values, key_weight, value_weight, key_bias,
            value_bias, query_mask, selection_fill, key_rope_encoding,
            key_positions, rope_freqs, index_tensor, is_specified_mask, attn_weights,
            embed_dim, n_heads,
            scale_factor, needed_grads, dropout_p, training, attn_dropout_mask,
        )
        # fmt: on

    @staticmethod
    def _determine_needed_intermediate_grads(
        needed_grads: dict[str, bool],
    ) -> dict[str, bool]:
        """Determine which intermediate tensors need to be computed to compute
        all needed gradients
        """
        # All upstream gradients require grad of attention scores
        compute_grad_attn_scores = (
            needed_grads["query"]
            or needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
            or needed_grads["selection_fill"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Query tensor is its own branch, so no other grads depend on it
        compute_grad_query = needed_grads["query"]

        # Everything else is upstream of keys
        compute_grad_k = (
            needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
            or needed_grads["selection_fill"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Decide whether we need to go into the key branch
        # (only gradient that doesn't depend on it is the value projection)
        compute_grads_key_branch = (
            needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["key_bias"]
            or needed_grads["selection_fill"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Do we need to compute the grads of the on-the-fly RoPE encoding
        compute_grads_rope_inputs = (
            needed_grads["key_positions"] or needed_grads["rope_freqs"]
        )

        # Do we need to traverse the value branch
        compute_grads_value_branch = (
            needed_grads["sparse_values"]
            or needed_grads["value_weight"]
            or needed_grads["value_bias"]
            or needed_grads["selection_fill"]
        )

        # Input projections
        compute_grads_input_projections = (
            needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
            or needed_grads["selection_fill"]
        )

        # sparse_tensor_values is upstream of everything so nothing depends on it
        compute_grads_sparse_values = (
            needed_grads["sparse_values"] or needed_grads["selection_fill"]
        )

        return {
            "attn_scores": compute_grad_attn_scores,
            "query": compute_grad_query,
            "keys": compute_grad_k,
            "key_branch": compute_grads_key_branch,
            "rope_inputs": compute_grads_rope_inputs,
            "value_branch": compute_grads_value_branch,
            "input_projections": compute_grads_input_projections,
            "sparse_values": compute_grads_sparse_values,
        }

    @staticmethod
    @torch.autograd.function.once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        """Implements the backward pass for sparse neighborhood attention.

        This custom backward operation recalculates intermediate values that were
        not stored during the forward pass to save memory, then calculates gradients
        for only the input tensors that require gradients.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context containing saved tensors
            grad_output (Tensor): Gradient of the loss with respect to the output,
                shape [n_queries, embed_dim]

        Returns:
            tuple[Optional[Tensor], ...]: Gradients for all inputs in the same order as
            the forward method:
                - grad_query: [n_queries, embed_dim] or None
                - None (for n_heads)
                - grad_sparse_values: [num_sparse_values, embed_dim] or None
                - None (for index_tensor)
                - None (for is_specified_mask)
                - grad_key_weight: [embed_dim, embed_dim] or None
                - grad_value_weight: [embed_dim, embed_dim] or None
                - grad_key_bias: [embed_dim] or None
                - grad_value_bias: [embed_dim] or None
                - None (for query_mask)
                - grad_selection_fill: Same dim as input `selection_fill`, or None
                - grad_key_rope_encoding: [n_queries, n_keys_per_query, embed_dim] or None
                - grad_key_positions: [n_queries, n_keys_per_query, position_dim] or None
                - grad_rope_freqs: [position_dim, n_freq_groups, n_heads, head_dim/2] or None
                - None (for scale_factor)
                - None (for dropout_p)
                - None (for training)
        """

        ######
        # The backward pass is broken into steps:
        # 1. Retrieve all inputs and set up variables
        # 2. Deduce which intermediate values need to be recalculated
        # 3. Repeat the first few steps of the forward pass
        #   - Input and key and value projections, and unstacking the heads of queries, keys, and values
        #   - Calculation of the RoPE encoding and applying it to the key tensor
        #   - Permutation of queries, keys, and values into batched-heads structure (as standard)
        # 4. If any gradients are required, compute the gradient of the attention scores
        #   as it is downstream of all of the input parameters which may need gradients
        # 5. If query_vector needs a gradient, compute its gradient
        # 6. If any gradients along the key branch (key projection, RoPE parameters,
        #   or the input sparse tensor values) are needed, compute the grad of keys
        # 7. If RoPE was applied, apply its backward to the grad of keys, and get the
        #   gradients of the RoPE encoding or its inputs (key positions and rope
        #   frequencies) if we need those
        # 8. If the value projection or the sparse tensor values need gradients,
        #   compute the grad of values
        # 9. If the input key and/or value projections need gradients, compute those
        # 10. Finally, if the sparse tensor values need gradients, compute those
        #
        # All intermediate tensors are deleted as soon as they aren't needed anymore
        #   (i.e., when there aren't any more tensors upstream of them to compute)
        ######

        ###### Implementation note ######
        # While this backward could potentially be better modularized by moving the
        #   multi-step gradient calculation to a helper function, this would prevent
        #   us from deleting tensors (keys, values, etc.) to free memory as soon as
        #   they are not needed, because there would still be a reference to them in
        #   the context of the main backward until it returns and we're able to
        #   release them there.
        # A future version of this code might work around this with a manager object
        #   that centrally holds all the references to every tensor and gets passed
        #   around, with the deletions acting on that object's references instead of
        #   the references in the backward namespace.
        #
        ######

        # Early return for no gradients
        if grad_output is None:
            return (None,) * 17

        ##### Step 1: retrieve values and set up variables

        # fmt: off
        (
            query_tensor, sparse_tensor_values, key_weight, value_weight, key_bias,
            value_bias, query_mask, selection_fill, key_rope_encoding,
            key_positions, rope_freqs, index_tensor, is_specified_mask, attn_weights,
            embed_dim, n_heads,
            scale_factor, needed_grads, dropout_p, training, attn_dropout_mask,
        ) = GatherAndSubsetAttentionFunction._initialize_backward(ctx)
        # fmt: on

        # initialize grad vars
        grad_query = None
        grad_sparse_values = None
        grad_key_weight = None
        grad_value_weight = None
        grad_key_bias = None
        grad_value_bias = None
        grad_selection_fill = None
        grad_key_rope_encoding = None
        grad_key_positions = None
        grad_rope_freqs = None

        # initialize flattened grad vars
        grad_keys_flat = None
        grad_values_flat = None

        # initialize pre-rotation keys copy for RoPE grads
        keys_unrotated_copy = None

        #### Step 2: Decide which which gradients to compute

        needed_intermediates = (
            GatherAndSubsetAttentionFunction._determine_needed_intermediate_grads(
                needed_grads
            )
        )

        #### Step 3: repeat the first few operations of the forward pass

        (
            queries,
            keys,
            values,
            key_rope_encoding,
            selected,
            keys_unrotated_copy,  # pyright: ignore[reportAssignmentType]
        ) = GatherAndSubsetAttentionFunction._forward_shared(
            query_tensor,
            sparse_tensor_values,
            index_tensor,
            is_specified_mask,
            key_weight,
            value_weight,
            key_bias,
            value_bias,
            selection_fill,
            n_heads,
            key_positions,
            rope_freqs,
            key_rope_encoding,
            is_for_backward=True,
            need_backward_key_branch=needed_intermediates["key_branch"],
        )

        grad_output = permute_for_attention(split_heads(grad_output, n_heads))

        #### Step 4: Compute gradient of attention scores

        if needed_intermediates["attn_scores"]:
            grad_attn_scores = _compute_grad_attn_scores(
                grad_output,
                values,
                attn_weights,
                is_specified_mask,
                dropout_p,
                training,
                selection_fill,
                attn_dropout_mask,
                query_mask,
            )
        del values  # big tensor we no longer need

        #### Step 5: Compute query gradient

        if needed_grads["query"]:
            grad_query = _compute_grad_query(grad_attn_scores, keys, scale_factor)
        del keys

        #### Step 6: Compute gradient of keys, representing the entry to the key branch

        if needed_intermediates["keys"]:
            if needed_intermediates["key_branch"]:
                #### Step 7: Compute the backward pass of RoPE and compute the RoPE
                #       encoding's gradient, if needed.

                # combining the computation of grad_keys and un-rotating it into one
                # function makes the main backward's logic less complex
                grad_keys, grad_key_rope_encoding = (
                    _compute_grads_keys_and_rope_encoding(
                        grad_attn_scores,
                        queries,
                        keys_unrotated_copy,
                        scale_factor,
                        key_rope_encoding,
                        needs_grad_k=(
                            needed_grads["key_weight"]
                            or needed_grads["key_bias"]
                            or needed_grads["sparse_values"]
                            or needed_grads["selection_fill"]
                        ),
                        needs_grad_key_pos=(
                            needed_grads["key_rope_encoding"]
                            or needed_grads["key_positions"]
                            or needed_grads["rope_freqs"]
                        ),
                    )
                )
                del keys_unrotated_copy

                #### Step 7.5: Compute the gradients of the RoPE encoding's inputs
                #       if RoPE was computed on the fly

                if needed_intermediates["rope_inputs"]:
                    assert not needed_grads["key_rope_encoding"]  # mutually exclusive
                    grad_key_positions, grad_rope_freqs = calculate_rope_backward(
                        grad_key_rope_encoding,
                        key_positions,
                        rope_freqs,
                        needed_grads["key_positions"],
                        needed_grads["rope_freqs"],
                    )
                    grad_key_rope_encoding = None

                # Flatten for grad calcs
                if grad_keys is not None:
                    # [n_queries * n_keys_per_query * n_heads, head_dim]
                    grad_keys_flat = grad_keys.view(-1, embed_dim)
            del grad_attn_scores
            del queries

            ##### Step 8: Enter value branch

            if needed_intermediates["value_branch"]:
                grad_values = _compute_grad_values(
                    attn_weights, grad_output, dropout_p, training, attn_dropout_mask
                )

                # Flatten for grad calcs
                # [n_queries * n_keys_per_query * n_heads, head_dim]
                grad_values_flat = grad_values.view(-1, embed_dim)

            ##### Step 9: Input projection gradients

            if needed_intermediates["input_projections"]:
                # need to get at least one of the projection gradients
                grad_key_weight, grad_value_weight, grad_key_bias, grad_value_bias = (
                    _compute_grads_key_value_projections(
                        grad_keys_flat,
                        grad_values_flat,
                        selected,
                        needed_grads["key_weight"],
                        needed_grads["value_weight"],
                        needed_grads["key_bias"],
                        needed_grads["value_bias"],
                    )
                )
            del selected

            ##### Step 10: Gradients of the original sparse tensor values

            if needed_grads["sparse_values"] or needed_grads["selection_fill"]:
                grad_sparse_values, grad_selection_fill = _compute_grad_sparse_values(
                    grad_keys_flat,
                    grad_values_flat,
                    key_weight,
                    value_weight,
                    is_specified_mask,
                    sparse_tensor_values,
                    index_tensor,
                    selection_fill,
                    needed_grads["sparse_values"],
                    needed_grads["selection_fill"],
                )

        return (
            grad_query,  # query_tensor
            None,  # n_heads
            grad_sparse_values,  # sparse_tensor_values
            None,  # index_tensor
            None,  # is_specified_mask
            grad_key_weight,  # key_weight
            grad_value_weight,  # value_weight
            grad_key_bias,  # key_bias
            grad_value_bias,  # value_bias
            None,  # query_mask
            grad_selection_fill,  # selection_fill
            grad_key_rope_encoding,  # key_rope_encoding
            grad_key_positions,  # key_positions
            grad_rope_freqs,  # rope_freqs
            None,  # scale_factor
            None,  # dropout_p
            None,  # training
        )


##### Helper functions for calculation steps


@torch.jit.script
def _compute_grad_attn_scores(
    grad_output: Tensor,
    values: Tensor,
    attn_weights: Tensor,
    is_specified_mask: Tensor,
    dropout_p: float,
    training: bool,
    selection_fill: Optional[Tensor],
    attn_dropout_mask: Optional[Tensor] = None,
    query_mask: Optional[Tensor] = None,
) -> Tensor:
    """Computes gradients of attention scores with respect to softmax attention weights.
    Implements the softmax gradient.
    """
    # fmt: off
    grad_attn_weights = torch.matmul(
        grad_output.unsqueeze(-2), # (n_heads, n_queries, 1, head_dim)
        values.transpose(-1, -2),  # (n_heads, n_queries, head_dim, n_keys_per_query)
    ).squeeze(-2)                  # (n_heads, n_queries, n_keys_per_query)
    # fmt: on

    # Apply dropout to attn_weights if it was done in the forward pass
    if training and dropout_p > 0.0:
        assert attn_dropout_mask is not None
        dropout_scale = 1.0 / (1.0 - dropout_p)
        attn_weights = attn_weights.masked_fill(attn_dropout_mask, 0.0) * dropout_scale

    # softmax gradient: dL/dz = S * (dL/dS - sum_j(S_j * dL/dS_j))
    # where z = attn_scores, S = softmax(z), dL/dS = grad_attn_weights
    # and j indexes keys
    grad_attn_scores = attn_weights * (
        grad_attn_weights - (attn_weights * grad_attn_weights).sum(-1, keepdim=True)
    )

    if selection_fill is None:
        grad_attn_scores.masked_fill_(~is_specified_mask, 0.0)

    if query_mask is not None:
        grad_attn_scores.masked_fill_(query_mask[None, :, None], 0.0)

    return grad_attn_scores


@torch.jit.script
def _compute_grad_query(grad_attn_scores: Tensor, keys: Tensor, scale_factor: float):
    """Computes query gradients by backpropagating through the attention score calculation."""
    # fmt: off
    grad_queries = torch.matmul(
        grad_attn_scores.unsqueeze(-2),  # (n_heads, n_queries, 1, n_keys_per_query)
        keys,                            # (n_heads, n_queries, n_keys_per_query, head_dim)
    ).squeeze(-2)                        # (n_heads, n_queries, head_dim
    # fmt: on

    grad_queries *= scale_factor

    # Flip dims back and stack heads
    # (n_queries, n_heads, head_dim)
    grad_queries = permute_for_attention_backward(grad_queries)

    grad_query = grad_queries.flatten(-2, -1)  # (n_queries, embed_dim)
    return grad_query


@torch.jit.script
def _compute_grads_keys_and_rope_encoding(
    grad_attn_scores: Tensor,
    queries: Tensor,
    keys: Tensor,
    scale_factor: float,
    key_rope_encoding: Union[Tensor, None],
    needs_grad_k: bool,
    needs_grad_key_pos: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Computes gradients for keys and rotary position encoding tensors."""
    if not needs_grad_k and not needs_grad_key_pos:
        return None, None
    # fmt: off
    grad_keys_maybe_rotated = torch.matmul(
        grad_attn_scores.unsqueeze(-1),  # (n_heads, n_queries, n_keys_per_query, 1)
        queries.unsqueeze(-2),           # (n_heads, n_queries, 1, head_dim)
    )                                    # (n_heads, n_queries, n_keys_per_query, head_dim)
    # fmt: on
    grad_keys_maybe_rotated *= scale_factor

    # (n_queries, n_keys_per_query, n_heads, head_dim)
    grad_keys_maybe_rotated = permute_for_attention_backward(grad_keys_maybe_rotated)

    if key_rope_encoding is not None:
        # Handle backpropagation through RoPE
        grad_keys, grad_key_rope_encoding = rotate_embeddings_backward(
            grad_keys_maybe_rotated,
            keys,
            key_rope_encoding,
            needs_grad_k,
            needs_grad_key_pos,
            needs_autograd=False,
        )
    else:
        grad_keys = grad_keys_maybe_rotated
        grad_key_rope_encoding = None

    if grad_keys is None:
        assert not needs_grad_k
        return None, grad_key_rope_encoding

    # (n_heads, n_queries, embed_dim)
    grad_keys = grad_keys.flatten(-2, -1)  # stack heads

    return grad_keys, grad_key_rope_encoding


@torch.jit.script
def _compute_grad_values(
    attn_weights: Tensor,
    grad_output: Tensor,
    dropout_p: float,
    training: bool,
    attn_dropout_mask: Optional[Tensor] = None,
) -> Tensor:
    """Computes value gradients by propagating output gradients through attention weights."""
    # Apply dropout to attn_weights if it was done in the forward pass
    if training and dropout_p > 0.0:
        assert attn_dropout_mask is not None
        dropout_scale = 1.0 / (1.0 - dropout_p)
        attn_weights = attn_weights.masked_fill(attn_dropout_mask, 0.0) * dropout_scale

    # fmt: off
    grad_values = torch.matmul(
        attn_weights.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
        grad_output.unsqueeze(-2)   # (n_heads, n_queries, 1, head_dim)
    )                               # (n_heads, n_queries, n_keys_per_query, head_dim)
    # fmt: on

    # (n_queries, n_keys_per_query, n_heads, head_dim)
    grad_values = permute_for_attention_backward(grad_values)

    # (n_queries, n_keys_per_query, embed_dim)
    grad_values = grad_values.flatten(-2, -1)
    return grad_values


@torch.jit.script
def _compute_grads_key_value_projections(
    grad_keys_flat: Optional[Tensor],
    grad_values_flat: Optional[Tensor],
    selected: Tensor,
    needs_grad_key_weight: bool,
    needs_grad_value_weight: bool,
    needs_grad_key_bias: bool,
    needs_grad_value_bias: bool,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Computes gradients for key and value projection weights and biases, with optional batching for efficiency."""
    selected_flat = selected.view(-1, selected.size(-1))

    if (needs_grad_key_weight or needs_grad_key_bias) and (
        needs_grad_value_weight and needs_grad_value_bias
    ):
        # need grads from both projections - batch the two gradient
        # calculations to save a matmul call (bmm vs 2x mm)

        # type refinement for torchscript
        assert grad_keys_flat is not None
        assert grad_values_flat is not None

        # stack gradients for batched keys and values backward (adding leading dim)
        grad_kv_flat = torch.stack([grad_keys_flat, grad_values_flat])

        grad_weights_stacked, grad_bias_stacked = linear_grads(
            grad_kv_flat,
            selected_flat,
            needs_grad_key_weight or needs_grad_value_weight,
            needs_grad_key_bias or needs_grad_value_bias,
        )

        if grad_weights_stacked is not None:
            grad_key_weight, grad_value_weight = grad_weights_stacked.unbind(0)
            grad_key_weight = grad_key_weight if needs_grad_key_weight else None
            grad_value_weight = grad_value_weight if needs_grad_value_weight else None
        else:
            grad_key_weight, grad_value_weight = None, None

        if grad_bias_stacked is not None:
            grad_key_bias, grad_value_bias = grad_bias_stacked.unbind(0)
            grad_key_bias = grad_key_bias if needs_grad_key_bias else None
            grad_value_bias = grad_value_bias if needs_grad_value_bias else None
        else:
            grad_key_bias, grad_value_bias = None, None

    else:
        # only need one projection's grad. call linear_grads twice
        # since it will safely return None, None for the one where
        # needs_grads bools are False
        grad_key_weight, grad_key_bias = linear_grads(
            grad_keys_flat,
            selected_flat,
            needs_grad_key_weight,
            needs_grad_key_bias,
        )

        grad_value_weight, grad_value_bias = linear_grads(
            grad_values_flat,
            selected_flat,
            needs_grad_value_weight,
            needs_grad_value_bias,
        )
    return grad_key_weight, grad_value_weight, grad_key_bias, grad_value_bias


@torch.jit.script
def _compute_grad_sparse_values(
    grad_keys_flat: Tensor,
    grad_values_flat: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    is_specified_mask: Tensor,
    sparse_tensor_values: Tensor,
    index_tensor: Tensor,
    selection_fill: Optional[Tensor],
    need_grad_sparse_values: bool,
    need_grad_selection_fill: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Computes gradients for sparse input values by applying projection weights and scattering back to sparse structure."""
    n_queries = is_specified_mask.size(0)
    n_keys_per_query = is_specified_mask.size(1)
    embed_dim = grad_keys_flat.size(-1)

    # two matrix multiplies - faster if we batch them
    grad_k_v_stacked = torch.stack([grad_keys_flat, grad_values_flat])
    W_stacked = torch.stack([key_weight, value_weight])
    # fmt: off
    grad_selected = torch.bmm(
        grad_k_v_stacked,  # (2, n_queries * n_keys_per_query, embed_dim)
        W_stacked,         # (2, embed_dim, embed_dim)
    )                      # (2, n_queries * n_keys_per_query, embed_dim)
    # fmt: on

    # elementwise add of keys, values contributions
    grad_selected = grad_selected.sum(0)

    grad_selected = grad_selected.view(n_queries, n_keys_per_query, embed_dim)

    grad_sparse_values: Optional[Tensor] = None
    grad_selection_fill: Optional[Tensor] = None

    # Scatter grads back into the sparse values
    if need_grad_sparse_values:
        grad_sparse_values = torch.zeros_like(sparse_tensor_values)
        grad_selected_specified = grad_selected.masked_fill(
            ~is_specified_mask.unsqueeze(-1), 0.0
        )
        grad_sparse_values.index_add_(
            0, index_tensor.view(-1), grad_selected_specified.view(-1, embed_dim)
        )
        # grad_sparse_values_2 = torch.sparse_coo_tensor(
        #     index_tensor.view(1, -1),
        #     grad_selected_specified.view(-1, embed_dim),
        #     sparse_tensor_values.shape,
        # ).coalesce()

        # # Verify gradient equality
        # grad_sparse_values_2_dense = grad_sparse_values_2.to_dense()
        # assert torch.equal(
        #     grad_sparse_values.nonzero(), grad_sparse_values_2_dense.nonzero()
        # )
        # nonzero_grad_sparse_values = grad_sparse_values[grad_sparse_values.sum(-1) > 0.0]
        # nonzero_grad_sparse_values_2 = grad_sparse_values_2_dense[
        #     grad_sparse_values_2_dense.sum(-1) > 0.0
        # ]
        # cos_sim = torch.cosine_similarity(
        #     nonzero_grad_sparse_values, nonzero_grad_sparse_values_2
        # )
        # assert (cos_sim > 0.9999).all()
        # grad_sparse_values_norm = torch.linalg.vector_norm(grad_sparse_values, dim=-1)
        # grad_sparse_values_2_norm = torch.linalg.vector_norm(grad_sparse_values_2_dense, dim=-1)
        # assert torch.allclose(grad_sparse_values_norm, grad_sparse_values_2_norm)

    if need_grad_selection_fill:
        assert selection_fill is not None
        grad_selected_unspecified = grad_selected.masked_fill(
            is_specified_mask.unsqueeze(-1), 0.0
        )
        grad_selection_fill = grad_selected_unspecified.sum_to_size(
            selection_fill.shape
        )

    return grad_sparse_values, grad_selection_fill

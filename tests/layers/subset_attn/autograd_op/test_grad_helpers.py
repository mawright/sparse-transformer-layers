import math
from typing import Union, Optional
import pytest
import torch
from hypothesis import strategies as st
from hypothesis import given, settings, HealthCheck
from torch import Tensor
import torch.nn.functional as F

from sparse_transformer_layers.layers.subset_attn.autograd import (
    _compute_grad_attn_scores,
    _compute_grad_query,
    _compute_grad_sparse_values,
    _compute_grad_values,
    _compute_grads_key_value_projections,
    _compute_grads_keys_and_rope_encoding,
)
from sparse_transformer_layers.layers.subset_attn.autograd_helpers import (
    permute_for_attention,
    split_heads,
)
from nd_rotary_encodings.functional import rotate_embeddings


def random_attn_tensors(
    n_queries: int = 3,
    n_keys_per_query: int = 4,
    n_heads: int = 4,
    head_dim: int = 6,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
):
    rng = torch.Generator(device)
    rng.manual_seed(seed)
    q = torch.randn(n_queries, head_dim * n_heads, device=device, generator=rng)
    k = torch.randn(
        n_queries, n_keys_per_query, head_dim * n_heads, device=device, generator=rng
    )
    v = torch.empty_like(k).normal_(generator=rng)

    scale_factor = 1 / math.sqrt(head_dim)

    return q, k, v, scale_factor


def assert_print_diff(result: Tensor, expected: Tensor, atol=1e-8, rtol=1e-5):
    abs_diff = (result - expected).abs()
    max_diff = abs_diff.max()
    max_rel_diff = (abs_diff / result).abs().max()

    assert torch.allclose(
        result, expected, atol=atol, rtol=rtol
    ), f"max diff={max_diff}, rel diff={max_rel_diff}"


@pytest.mark.cuda_if_available
class TestComputeGradAttnScores:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_heads=st.integers(1, 5),
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        head_dim=st.integers(1, 8),
        dropout_p=st.floats(0.0, 1.0, exclude_max=True),
        training=st.booleans(),
        use_query_mask=st.booleans(),
        query_mask_prob=st.floats(0.0, 1.0, exclude_max=True),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_hypothesis(
        self,
        n_heads: int,
        n_q: int,
        n_k: int,
        head_dim: int,
        dropout_p: float,
        training: bool,
        use_query_mask: bool,
        query_mask_prob: float,
        seed: int,
        device,
    ):
        rng = torch.Generator(device)
        rng.manual_seed(seed)

        q, k, v, scale = random_attn_tensors(n_q, n_k, n_heads, head_dim, seed, device)

        q = permute_for_attention(split_heads(q, n_heads))
        k = permute_for_attention(split_heads(k, n_heads))
        v = permute_for_attention(split_heads(v, n_heads))

        attn_scores = torch.matmul(
            (q.unsqueeze(-2) * scale), k.transpose(-1, -2)
        ).squeeze(
            -2
        )  # (h, q, k)
        attn_scores_leaf = attn_scores.detach().clone().requires_grad_(True)
        attn_scores = attn_scores_leaf

        query_mask = None
        if use_query_mask and query_mask_prob > 0.0:
            query_mask = torch.empty(n_q, device=device, dtype=torch.bool)
            query_mask.bernoulli_(query_mask_prob, generator=rng)
            attn_weights = torch.where(
                query_mask[None, :, None],
                torch.zeros_like(attn_scores),
                attn_scores.softmax(-1),
            )
        else:
            attn_weights = attn_scores.softmax(-1)

        attn_weights_copy = attn_weights.detach().clone()

        attn_dropout_mask = None
        if training and dropout_p > 0.0:
            # Create and apply dropout mask
            attn_dropout_mask = torch.empty_like(attn_weights, dtype=torch.bool)
            attn_dropout_mask.bernoulli_(dropout_p)  # 1 means drop this element
            dropout_scale = 1.0 / (1.0 - dropout_p)

            attn_weights_dropped = attn_weights.clone()
            attn_weights_dropped = attn_weights_dropped.masked_fill(
                attn_dropout_mask, 0.0
            )
            attn_weights_dropped = attn_weights_dropped * dropout_scale

            attn_weights = attn_weights_dropped

        # Compute grad with autograd
        output = torch.matmul(attn_weights.unsqueeze(-2), v).squeeze(-2)  # (h, q, d)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        expected = attn_scores_leaf.grad
        assert expected is not None

        # Compute grad with helper
        grad_scores = _compute_grad_attn_scores(
            grad_output,
            v,
            attn_weights_copy,
            torch.ones(n_q, n_k, dtype=torch.bool, device=device),
            dropout_p=dropout_p,
            training=training,
            selection_fill=None,
            attn_dropout_mask=attn_dropout_mask,
            query_mask=None,
        )

        assert torch.allclose(
            expected, grad_scores, atol=1e-6
        ), f"max diff: {(expected - grad_scores).abs().max()}"


@pytest.mark.cuda_if_available
class TestComputeGradQuery:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_heads=st.integers(1, 5),
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        head_dim=st.integers(1, 8),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_hypothesis(
        self,
        n_heads: int,
        n_q: int,
        n_k: int,
        head_dim: int,
        seed: int,
        device,
    ):
        q, k, _, scale = random_attn_tensors(n_q, n_k, n_heads, head_dim, seed, device)
        q_leaf = q.requires_grad_(True)

        q = permute_for_attention(split_heads(q, n_heads))
        k = permute_for_attention(split_heads(k, n_heads))

        attn_scores = torch.matmul(
            q.unsqueeze(-2) * scale, k.transpose(-1, -2)
        ).squeeze(-2)
        grad_attn_scores = torch.randn_like(attn_scores)

        attn_scores.backward(grad_attn_scores)

        expected = q_leaf.grad
        assert expected is not None

        grad_q = _compute_grad_query(grad_attn_scores, k, scale)

        assert torch.allclose(
            expected, grad_q
        ), f"max diff: {(expected - grad_q).abs().max()}"


@pytest.mark.cuda_if_available
class TestComputeGradKeysAndRoPEEncoding:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_heads=st.integers(1, 5),
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        head_dim=st.integers(1, 8).filter(lambda x: x % 2 == 0),
        include_rope_encoding=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_hypothesis(
        self,
        n_heads: int,
        n_q: int,
        n_k: int,
        head_dim: int,
        include_rope_encoding: bool,
        seed: int,
        device,
    ):
        queries, keys, _, scale = random_attn_tensors(
            n_q, n_k, n_heads, head_dim, seed, device
        )

        keys_leaf = keys.requires_grad_(True)

        queries: Tensor = split_heads(queries, n_heads)  # (q, h, d)
        keys: Tensor = split_heads(keys, n_heads)  # (q, k, h, d)
        keys_unpermuted = keys.detach().clone()

        rope_encoding = None
        if include_rope_encoding:
            rope_encoding = torch.randn(
                n_q,
                n_k,
                n_heads,
                head_dim // 2,
                device=device,
                dtype=keys.dtype,
                requires_grad=True,
            )
            keys = rotate_embeddings(keys, rope_encoding, needs_autograd=True)

        queries: Tensor = permute_for_attention(queries)  # (h, q, d)
        keys: Tensor = permute_for_attention(keys)  # (h, q, k, d)

        attn_scores = torch.matmul(
            queries.unsqueeze(-2) * scale, keys.transpose(-1, -2)
        ).squeeze(
            -2
        )  # (h, q, k)
        grad_attn_scores = torch.randn_like(attn_scores)

        attn_scores.backward(grad_attn_scores)

        assert keys_leaf.grad is not None
        if rope_encoding is not None:
            assert rope_encoding.grad is not None

        grad_keys, grad_rope_encoding = _compute_grads_keys_and_rope_encoding(
            grad_attn_scores,
            queries,
            keys_unpermuted,
            scale,
            rope_encoding,
            True,
            include_rope_encoding,
        )

        abs_diff = (grad_keys - keys_leaf.grad).abs()
        max_diff = abs_diff.max()
        max_rel_diff = (abs_diff / grad_keys).abs()

        assert torch.allclose(
            grad_keys, keys_leaf.grad, atol=1e-7
        ), f"max diff={max_diff}, max rel diff={max_rel_diff}"
        if grad_rope_encoding is not None:
            assert rope_encoding is not None
            assert rope_encoding.grad is not None
            abs_diff = (grad_rope_encoding - rope_encoding.grad).abs()
            max_diff = abs_diff.max()
            max_rel_diff = (abs_diff / grad_rope_encoding).abs()
            assert torch.allclose(
                grad_rope_encoding, rope_encoding.grad, atol=1e-7
            ), f"max diff={max_diff}, max rel diff={max_rel_diff}"
        else:
            assert rope_encoding is None


class TestComputeGradValues:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_heads=st.integers(1, 5),
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        head_dim=st.integers(1, 8).filter(lambda x: x % 2 == 0),
        seed=st.integers(0, 2**32 - 1),
        training=st.booleans(),
        dropout_p=st.floats(0.0, 1.0, exclude_max=True),
    )
    def test_hypothesis(
        self,
        n_heads: int,
        n_q: int,
        n_k: int,
        head_dim: int,
        seed: int,
        training: bool,
        dropout_p: float,
        device,
    ):
        _, _, values, _ = random_attn_tensors(n_q, n_k, n_heads, head_dim, seed, device)
        values_leaf = values.requires_grad_(True)

        values = permute_for_attention(split_heads(values, n_heads))

        attn_weights = torch.randn(n_heads, n_q, n_k, device=device, dtype=values.dtype)
        attn_weights_copy = attn_weights.clone()

        attn_dropout_mask = None
        if training and dropout_p > 0.0:
            # Create and apply dropout mask
            attn_dropout_mask = torch.empty_like(attn_weights, dtype=torch.bool)
            attn_dropout_mask.bernoulli_(dropout_p)  # 1 means drop this element
            dropout_scale = 1.0 / (1.0 - dropout_p)

            attn_weights = (
                attn_weights.masked_fill(attn_dropout_mask, 0.0) * dropout_scale
            )

        output = torch.matmul(
            attn_weights.unsqueeze(-2),
            values,
        ).squeeze(-2)

        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        assert values_leaf.grad is not None

        result = _compute_grad_values(
            attn_weights_copy, grad_output, dropout_p, training, attn_dropout_mask
        )

        abs_diff = (result - values_leaf.grad).abs()
        max_diff = abs_diff.max()
        max_rel_diff = (abs_diff / result).abs()

        assert torch.allclose(
            result, values_leaf.grad
        ), f"max diff={max_diff}, max rel diff={max_rel_diff}"


class TestComputeGradsKeyValueProjections:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_heads=st.integers(1, 5),
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        head_dim=st.integers(1, 8).filter(lambda x: x % 2 == 0),
        grad_key_weight=st.booleans(),
        grad_value_weight=st.booleans(),
        grad_key_bias=st.booleans(),
        grad_value_bias=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_hypothesis(
        self,
        n_heads: int,
        n_q: int,
        n_k: int,
        head_dim: int,
        grad_key_weight: bool,
        grad_value_weight: bool,
        grad_key_bias: bool,
        grad_value_bias: bool,
        seed: int,
        device,
    ):
        torch.manual_seed(seed)
        embed_dim = head_dim * n_heads
        selected = torch.randn(n_q, n_k, embed_dim, device=device)

        Wk = torch.randn(
            embed_dim, embed_dim, device=device, requires_grad=grad_key_weight
        )
        Wv = torch.randn_like(Wk, requires_grad=grad_value_weight)

        bias_k = torch.randn(embed_dim, device=device, requires_grad=grad_key_bias)
        bias_v = torch.randn_like(bias_k, requires_grad=grad_value_bias)

        k = F.linear(selected, Wk, bias_k)
        v = F.linear(selected, Wv, bias_v)

        grad_k = torch.randn_like(k)
        grad_v = torch.randn_like(v)

        if k.requires_grad:
            k.backward(grad_k)

        if v.requires_grad:
            v.backward(grad_v)

        grad_Wk, grad_Wv, grad_bias_k, grad_bias_v = (
            _compute_grads_key_value_projections(
                grad_k.view(-1, embed_dim),
                grad_v.view(-1, embed_dim),
                selected,
                grad_key_weight,
                grad_value_weight,
                grad_key_bias,
                grad_value_bias,
            )
        )

        if grad_key_weight:
            assert Wk.grad is not None
            assert_print_diff(grad_Wk, Wk.grad, atol=1e-6)
        if grad_value_weight:
            assert Wv.grad is not None
            assert_print_diff(grad_Wv, Wv.grad, atol=1e-6)
        if grad_key_bias:
            assert bias_k.grad is not None
            assert_print_diff(grad_bias_k, bias_k.grad, atol=1e-6)
        if grad_value_bias:
            assert bias_v.grad is not None
            assert_print_diff(grad_bias_v, bias_v.grad, atol=1e-6)


@pytest.mark.cuda_if_available
class TestComputeGradSparseValues:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_q=st.integers(1, 4),
        n_k=st.integers(1, 4),
        embed_dim=st.integers(1, 8),
        include_selection_fill=st.booleans(),
        selection_fill_shape=st.sampled_from(["vector", "broadcast", "full"]),
        need_grad_sparse_values=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_hypothesis(
        self,
        n_q: int,
        n_k: int,
        embed_dim: int,
        include_selection_fill: bool,
        selection_fill_shape: str,
        need_grad_sparse_values: bool,
        seed: int,
        device,
    ):
        torch.manual_seed(seed)

        n_sparse = max(n_q * n_k, 5)

        # sparse tensor values (leaf)
        sparse_vals = torch.randn(
            n_sparse, embed_dim, device=device, requires_grad=True
        )

        # random index tensor and is_specified mask
        index_tensor = torch.randint(
            0, n_sparse, (n_q, n_k), device=device, dtype=torch.long
        )
        is_specified_mask = torch.empty(n_q, n_k, dtype=torch.bool, device=device)
        is_specified_mask.bernoulli_(p=0.7)

        # optional selection-fill tensor – choose a broadcast-friendly shape
        selection_fill: Optional[Tensor] = None
        if include_selection_fill:
            if selection_fill_shape == "vector":
                # vector (embed_dim,)
                sel_shape = (embed_dim,)
            elif selection_fill_shape == "broadcast":
                # [1, 1, embed_dim]
                sel_shape = (1, 1, embed_dim)
            else:
                # full [n_q, n_k, embed_dim]
                sel_shape = (n_q, n_k, embed_dim)
            selection_fill = torch.randn(sel_shape, device=device, requires_grad=True)

        # random linear projections
        key_weight = torch.randn(embed_dim, embed_dim, device=device)
        value_weight = torch.randn_like(key_weight)

        selected = sparse_vals[index_tensor]  # gather
        if selection_fill is not None:
            selected = torch.where(
                is_specified_mask.unsqueeze(-1), selected, selection_fill
            )
        else:
            selected = torch.where(
                is_specified_mask.unsqueeze(-1), selected, torch.zeros_like(selected)
            )

        # forward pass
        keys = F.linear(selected, key_weight)
        values = F.linear(selected, value_weight)

        # pretend upstream gradients
        grad_keys = torch.randn_like(keys)
        grad_values = torch.randn_like(values)

        # run autograd
        keys.backward(grad_keys, retain_graph=True)
        values.backward(grad_values)

        expected_sparse_grad = None
        expected_sel_fill_grad = None
        if need_grad_sparse_values:
            assert sparse_vals.grad is not None
            expected_sparse_grad = sparse_vals.grad.clone()
        if selection_fill is not None:
            assert selection_fill.grad is not None
            expected_sel_fill_grad = selection_fill.grad.clone()

        grad_sparse_values, grad_selection_fill = _compute_grad_sparse_values(
            grad_keys.view(-1, embed_dim),
            grad_values.view(-1, embed_dim),
            key_weight,
            value_weight,
            is_specified_mask,
            sparse_vals.detach(),
            index_tensor,
            selection_fill.detach() if selection_fill is not None else None,
            need_grad_sparse_values=need_grad_sparse_values,
            need_grad_selection_fill=selection_fill is not None,
        )

        if need_grad_sparse_values:
            assert expected_sparse_grad is not None
            assert_print_diff(grad_sparse_values, expected_sparse_grad, atol=1e-7)
        else:
            assert grad_sparse_values is None

        if selection_fill is not None:
            assert expected_sel_fill_grad is not None
            assert_print_diff(grad_selection_fill, expected_sel_fill_grad, atol=1e-7)
        else:
            assert grad_selection_fill is None

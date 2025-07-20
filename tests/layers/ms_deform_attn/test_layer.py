from typing import Any, Optional, Union

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytorch_sparse_utils.batching import (
    seq_lengths_to_batch_offsets,
)
from torch import Tensor

from sparse_transformer_layers.layers.sparse_ms_deform_attn.layer import (
    SparseMSDeformableAttention,
)

from ...conftest import (
    random_multilevel_sparse_tensor_indices,
)


@st.composite
def ms_deform_attn_strategy(draw: st.DrawFn, require_grads: bool = False):
    """Draws a set of arguments for SparseMSDeformableAttention init and forward."""
    # Module params
    embed_dim = draw(st.integers(16, 128).filter(lambda v: v % 8 == 0))
    n_heads = draw(st.integers(1, 8).filter(lambda h: embed_dim % h == 0))
    n_levels = draw(st.integers(1, 4))
    n_points = draw(st.integers(1, 4))

    # Data size parameters
    seq_lengths = draw(st.lists(st.integers(0, 32), min_size=1, max_size=4))
    batch_size = len(seq_lengths)
    query_batch_offsets = np.zeros((batch_size + 1,), dtype=int)
    query_batch_offsets[1:] = np.cumsum(seq_lengths)

    # monotonically increasing spatial shapes
    level_spatial_shapes = []
    last_h, last_w = 1, 1
    for _ in range(n_levels):
        h = draw(st.integers(last_h, last_h * 4 if last_h < 64 else last_h))
        w = draw(st.integers(last_w, last_w * 4 if last_w < 64 else last_w))
        last_h, last_w = h, w
        level_spatial_shapes.append((h, w))
    level_spatial_shapes = np.asarray(level_spatial_shapes, dtype=np.int64)

    sparsity = draw(st.floats(0.4, 1.0, exclude_max=True))
    use_background_embedding = draw(st.booleans())

    seed = draw(st.integers(0, 10**9))

    return {
        "embed_dim": embed_dim,
        "n_heads": n_heads,
        "n_levels": n_levels,
        "n_points": n_points,
        "seq_lengths": seq_lengths,
        "level_spatial_shapes": level_spatial_shapes,
        "sparsity": sparsity,
        "require_grads": require_grads,
        "use_background_embedding": use_background_embedding,
        "seed": seed,
    }


def build_random_inputs(
    embed_dim: int,
    n_heads: int,
    n_levels: int,
    n_points: int,
    seq_lengths: list[int],
    level_spatial_shapes: Union[np.ndarray, Tensor],
    sparsity: float,
    require_grads: bool,
    use_background_embedding: bool,
    seed: int,
    device: Union[str, torch.device],
) -> tuple[
    SparseMSDeformableAttention,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Optional[Tensor],
]:
    """
    Builds an instance of the SparseMSDeformableAttention module and its input arguments.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    # Make module
    module = SparseMSDeformableAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_levels=n_levels,
        n_points=n_points,
    ).to(device)

    n_queries = sum(seq_lengths)

    # Make query tensor
    query = torch.randn(
        n_queries, embed_dim, device=device, requires_grad=require_grads
    )

    query_spatial_positions = torch.rand(n_queries, 2, device=device)
    query_level_indices = torch.randint(0, n_levels, (n_queries,), device=device)

    # scale the positions inside their native level
    level_spatial_shapes = torch.as_tensor(level_spatial_shapes, device=device)
    query_spatial_positions *= level_spatial_shapes[query_level_indices]

    query_batch_offsets = seq_lengths_to_batch_offsets(
        torch.as_tensor(seq_lengths, dtype=torch.long, device=device)
    )
    assert isinstance(query_batch_offsets, Tensor)

    # sparse feature map
    max_h, max_w = level_spatial_shapes.max(0).values
    max_h, max_w = int(max_h), int(max_w)
    batch_size = len(seq_lengths)

    # randomly sample coordinate indices for the sparse feature map
    indices = random_multilevel_sparse_tensor_indices(
        level_spatial_shapes,
        sparsity,
        batch_size,
        max_nnz=2000,
        device=device,
    )
    values = torch.randn(indices.size(1), embed_dim, device=device)
    stacked_feature_maps = torch.sparse_coo_tensor(
        indices,
        values,
        size=[batch_size, max_h, max_w, n_levels, embed_dim],
        device=device,
    ).coalesce()
    if require_grads:
        stacked_feature_maps.requires_grad_(True)

    # make background embedding
    background_embedding = None
    if use_background_embedding:
        background_embedding = torch.randn(
            batch_size,
            n_levels,
            embed_dim,
            device=device,
            requires_grad=require_grads,
        )

    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state)
    else:
        torch.set_rng_state(rng_state)

    return (
        module,
        query,
        query_spatial_positions,
        query_batch_offsets,
        stacked_feature_maps,
        level_spatial_shapes,
        query_level_indices,
        background_embedding,
    )


@pytest.mark.cpu_and_cuda
@settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
@given(cfg=ms_deform_attn_strategy(require_grads=True))
def test_forward_shapes_and_grads(cfg: dict[str, Any], device: str | torch.device):
    """Test basic forward-backward usage."""
    (
        module,
        query,
        query_spatial_positions,
        query_batch_offsets,
        stacked_feature_maps,
        level_spatial_shapes,
        query_level_indices,
        background_embedding,
    ) = build_random_inputs(**cfg, device=device)

    out = module(
        query=query,
        query_spatial_positions=query_spatial_positions,
        query_batch_offsets=query_batch_offsets,
        stacked_feature_maps=stacked_feature_maps,
        level_spatial_shapes=level_spatial_shapes,
        query_level_indices=query_level_indices,
        background_embedding=background_embedding,
    )

    assert out.shape == query.shape
    assert out.dtype == query.dtype

    loss = out.sum()
    loss.backward()

    # parameters
    for name, p in module.named_parameters():
        assert p.grad is not None, f"parameter {name} did not receive gradient"

    # inputs that should get grad
    assert query.grad is not None
    assert stacked_feature_maps.grad is not None
    if background_embedding is not None:
        assert background_embedding.grad is not None


@pytest.mark.cpu_and_cuda
@settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
@given(cfg=ms_deform_attn_strategy())
def test_query_permutation_invariance(cfg: dict[str, Any], device: str | torch.device):
    """Tests permutation invariance of queries."""
    (
        module,
        query,
        query_spatial_positions,
        query_batch_offsets,
        stacked_feature_maps,
        level_spatial_shapes,
        query_level_indices,
        background_embedding,
    ) = build_random_inputs(**cfg, device=device)

    # baseline
    out = module(
        query=query,
        query_spatial_positions=query_spatial_positions,
        query_batch_offsets=query_batch_offsets,
        stacked_feature_maps=stacked_feature_maps,
        level_spatial_shapes=level_spatial_shapes,
        query_level_indices=query_level_indices,
        background_embedding=background_embedding,
    )

    # permute queries within batch items
    seq_lengths: list[int] = cfg["seq_lengths"]
    batch_size = len(seq_lengths)
    permuted_indices = []
    for b in range(batch_size):
        batch_start = query_batch_offsets[b]
        perm_b = torch.randperm(seq_lengths[b], device=device) + batch_start
        permuted_indices.append(perm_b)

    permuted_indices = torch.cat(permuted_indices)

    query_2 = query.detach().clone()[permuted_indices].requires_grad_(True)
    query_spatial_positions_2 = query_spatial_positions[permuted_indices]
    query_level_indices_2 = query_level_indices[permuted_indices]

    assert query.shape == query_2.shape

    out_2 = module(
        query=query_2,
        query_spatial_positions=query_spatial_positions_2,
        query_batch_offsets=query_batch_offsets,
        stacked_feature_maps=stacked_feature_maps,
        level_spatial_shapes=level_spatial_shapes,
        query_level_indices=query_level_indices_2,
        background_embedding=background_embedding,
    )
    # un-permute
    out_2_unperm = out_2[permuted_indices.argsort()]

    assert torch.allclose(out, out_2_unperm, atol=1e-6)


@pytest.mark.cpu_and_cuda
def test_zero_queries(device: str | torch.device):
    """Test with 0-length query tensor"""
    device = torch.device(device)
    embed_dim, n_heads, n_levels, n_points = 32, 4, 2, 4
    layer = SparseMSDeformableAttention(embed_dim, n_heads, n_levels, n_points).to(
        device
    )

    # no queries
    query = torch.empty(0, embed_dim, device=device)
    pos = torch.empty(0, 2, device=device)
    batch_offsets = torch.tensor([0, 0], dtype=torch.long, device=device)

    # make dummy sparse feature map with at least one entry
    idx = torch.tensor([[0], [0], [0], [0]], device=device, dtype=torch.long)
    val = torch.randn(1, embed_dim, device=device)
    feat = torch.sparse_coo_tensor(
        idx, val, size=[1, 1, 1, n_levels, embed_dim], device=device
    ).coalesce()
    lvl_shapes = torch.tensor([[1, 1], [1, 1]], device=device)

    out = layer(query, pos, batch_offsets, feat, lvl_shapes)
    assert out.shape == (0, embed_dim)


@pytest.mark.cpu_and_cuda
def test_single_point_simple_interp(device: str | torch.device):
    """
    See if we can recover a straightforward interpolation with a single level and
    with 0 offset.
    """
    device = torch.device(device)
    embed_dim, n_heads, n_levels, n_points = 16, 4, 1, 4
    layer = SparseMSDeformableAttention(embed_dim, n_heads, n_levels, n_points).to(
        device
    )

    with torch.no_grad():
        # force offsets to 0
        layer.sampling_offsets.weight.zero_()
        layer.sampling_offsets.bias.zero_()
        # make value and output projections identity
        Identity = torch.eye(embed_dim, device=device)
        layer.value_proj.weight.copy_(Identity)
        layer.value_proj.bias.zero_()
        layer.output_proj.weight.copy_(Identity)
        layer.output_proj.bias.zero_()

    # build a dense 2x2 feature map so the expected value is easy to compute
    dense = torch.randn(1, 2, 2, 1, embed_dim, device=device)
    sparse_maps = dense.to_sparse(dense_dim=1).coalesce()

    lvl_shapes = torch.tensor([[2, 2]], dtype=torch.long, device=device)

    # one query located at center of the 2x2 grid – value = 0
    query = torch.randn(1, embed_dim, device=device)
    spatial_positions = torch.tensor([[1.0, 1.0]], device=device)
    query_batch_offsets = torch.tensor([0, 1], dtype=torch.long, device=device)

    out = layer(query, spatial_positions, query_batch_offsets, sparse_maps, lvl_shapes)
    expected = dense.view(-1, embed_dim).mean(0)  # exact mean
    assert torch.allclose(out.squeeze(0), expected)

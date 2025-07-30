from torch import Tensor

import pytest
from hypothesis import strategies as st
from hypothesis import given, settings, HealthCheck

from ..layers.ms_deform_attn.test_layer import (
    ms_deform_attn_strategy,
    build_random_inputs,
)

from sparse_transformer_layers.blocks.ms_deform_attn import (
    SparseMSDeformableAttentionBlock,
)


@pytest.mark.cuda_if_available
@settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
@given(
    cfg=ms_deform_attn_strategy(require_grads=True),
    dropout=st.floats(0, 1.0),
    bias=st.booleans(),
    norm_first=st.booleans(),
)
def test_ms_deform_attn_block(
    cfg, dropout: float, bias: bool, norm_first: bool, device
):
    block = SparseMSDeformableAttentionBlock(
        cfg["embed_dim"],
        cfg["n_heads"],
        cfg["n_levels"],
        cfg["n_points"],
        dropout=dropout,
        bias=bias,
        norm_first=norm_first,
    ).to(device)

    (
        _,
        query,
        query_spatial_positions,
        query_batch_offsets,
        stacked_feature_maps,
        level_spatial_shapes,
        query_level_indices,
        background_embedding,
    ) = build_random_inputs(**cfg, device=device)

    out = block(
        query,
        query_spatial_positions,
        query_batch_offsets,
        stacked_feature_maps,
        level_spatial_shapes,
        background_embedding,
        query_level_indices
    )

    assert isinstance(out, Tensor)

    loss = out.sum()
    loss.backward()

    assert query.grad is not None
    assert stacked_feature_maps.grad is not None
    if background_embedding is not None:
        assert background_embedding.grad is not None

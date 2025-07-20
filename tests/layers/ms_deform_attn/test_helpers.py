import random
from typing import Any, Optional, Union

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from torch import Tensor

from sparse_transformer_layers.layers.sparse_ms_deform_attn.utils import (
    _make_index_and_weight_tensors,
    multilevel_sparse_bilinear_grid_sample,
    sparse_split_heads,
)
from pytorch_sparse_utils.batching import (
    seq_lengths_to_indices,
)
from pytorch_sparse_utils.indexing import batch_sparse_index
from ...conftest import random_multilevel_sparse_tensor_indices

from ...conftest import (
    simple_sparse_input_tensors,
)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for SparseMSDeformableAttention"""
    return {
        "embed_dim": 64,
        "n_heads": 4,
        "n_levels": 3,
        "n_points": 4,
    }


@st.composite
def grid_sample_strategy(draw, require_grads: bool = False):
    # Draw shape parameters
    n_heads = draw(st.integers(1, 8))
    head_dim = draw(st.integers(1, 16)) * 2
    embed_dim = n_heads * head_dim
    position_dim = draw(st.just(2))
    n_levels = draw(st.integers(1, 4))

    extra_batch_dims = draw(st.lists(st.integers(0, 10), min_size=0, max_size=3))

    # Draw data size parameters
    seq_lengths = draw(st.lists(st.integers(0, 32), min_size=1, max_size=4))
    batch_size = len(seq_lengths)
    query_batch_offsets = np.zeros((batch_size + 1,), dtype=int)
    query_batch_offsets[1:] = np.cumsum(seq_lengths)

    # Level shapes
    level_spatial_shapes = []
    last_level = [1] * position_dim
    for level in range(n_levels):
        shape = []
        for pos_dim in range(position_dim):
            shape.append(draw(st.integers(last_level[pos_dim], 1000 * (level + 1))))
        last_level = shape
        level_spatial_shapes.append(shape)

    level_spatial_shapes = np.array(level_spatial_shapes)
    assert np.array_equal(np.sort(level_spatial_shapes, 0), level_spatial_shapes)

    sparsity = draw(st.floats(0.4, 1.0, exclude_max=True))
    make_level_indices = draw(st.booleans())
    make_head_indices = draw(st.booleans())
    make_background_embedding = draw(st.booleans())

    seed = draw(st.integers(0, int(1e8)))

    return {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "embed_dim": embed_dim,
        "n_levels": n_levels,
        "extra_batch_dims": extra_batch_dims,
        "seq_lengths": seq_lengths,
        "level_spatial_shapes": level_spatial_shapes,
        "sparsity": sparsity,
        "require_grads": require_grads,
        "make_level_indices": make_level_indices,
        "make_head_indices": make_head_indices,
        "make_background_embedding": make_background_embedding,
        "seed": seed,
    }


def grid_sample_tensors(
    n_heads: int,
    head_dim: int,
    n_levels: int,
    extra_batch_dims: list[int],
    seq_lengths: list[int],
    level_spatial_shapes: Union[np.ndarray, Tensor],
    sparsity: float,
    require_grads: bool,
    make_level_indices: bool,
    make_head_indices: bool,
    make_background_embedding: bool,
    seed: int,
    device: Union[str, torch.device],
    **kwargs,
) -> dict[str, Optional[Tensor]]:
    if isinstance(device, str):
        device = torch.device(device, 0)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    batch_size = len(seq_lengths)
    sum_seq_lens = sum(seq_lengths)

    batch_indices = (
        seq_lengths_to_indices(torch.as_tensor(seq_lengths, device=device))
        .view([sum_seq_lens] + [1] * len(extra_batch_dims))
        .expand([sum_seq_lens] + extra_batch_dims)
    )
    if make_level_indices:
        level_indices = torch.arange(n_levels, device=device)
    else:
        level_indices = None
    if make_head_indices:
        head_indices = torch.arange(n_heads, device=device)
    else:
        head_indices = None

    level_spatial_shapes = torch.as_tensor(level_spatial_shapes, device=device)

    spatial_positions = torch.rand(
        [sum_seq_lens] + extra_batch_dims + [n_levels, n_heads, 2], device=device
    )
    if level_indices is not None:
        spatial_positions *= level_spatial_shapes[level_indices].unsqueeze(1)
    else:
        spatial_positions *= level_spatial_shapes.unsqueeze(1)

    if make_background_embedding:
        background_embedding = torch.randn(
            batch_size,
            n_levels,
            n_heads,
            head_dim,
            device=device,
            requires_grad=require_grads,
        )
    else:
        background_embedding = None

    # find max spatial shape
    max_spatial_shape = level_spatial_shapes.max(-2)[0]
    # if different spatial shapes per batch, find max among batch images
    if max_spatial_shape.ndim == 2:
        max_spatial_shape = max_spatial_shape.max(0)[0]
    assert max_spatial_shape.numel() == 2

    # make sparse tensor
    sparse_tensor_indices = random_multilevel_sparse_tensor_indices(
        level_spatial_shapes, sparsity, batch_size, 1000, device
    )
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor_indices,
        torch.randn(sparse_tensor_indices.size(1), head_dim * n_heads, device=device),
        size=[batch_size] + max_spatial_shape.tolist() + [n_levels, head_dim * n_heads],
        device=device,
    ).coalesce()

    # Check validity of the sparse tensor
    assert (sparse_tensor.indices() >= 0).all()
    for level in range(n_levels):
        level_mask = sparse_tensor.indices()[-1] == level
        sparse_level_indices = sparse_tensor.indices().T[level_mask, 1:-1]
        assert torch.all(sparse_level_indices < level_spatial_shapes[level])

    sparse_tensor: Tensor = sparse_split_heads(sparse_tensor, n_heads)

    if require_grads:
        sparse_tensor.requires_grad_(True)

    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state)
    else:
        torch.set_rng_state(rng_state)

    return {
        "sparse_tensor": sparse_tensor,
        "spatial_positions": spatial_positions,
        "batch_indices": batch_indices,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "head_indices": head_indices,
        "background_embedding": background_embedding,
    }


def make_random_sampling_points(
    sparse_tensor: Tensor,
    level_spatial_shapes: Tensor,
    n_pts: int,
    level_indices: Optional[Tensor] = None,
    head_indices: Optional[Tensor] = None,
):
    """Create a random batch of sampling positions for multilevel_sparse_bilinear_grid_sample."""
    device = sparse_tensor.device

    batch_size, _, _, n_levels, n_heads, _ = sparse_tensor.shape

    if level_indices is None:
        level_indices = torch.arange(n_levels, device=device)
    if head_indices is None:
        head_indices = torch.arange(n_heads, device=device)

    assert level_indices.ndim == 1
    assert head_indices.ndim == 1

    L = level_indices.numel()
    H = head_indices.numel()

    # random batch indices for every point
    batch_indices = torch.randint(0, batch_size, (n_pts,), device=device)

    level_shapes = level_spatial_shapes[level_indices]
    normalized_rand_pos = torch.rand(n_pts, L, H, 2, device=device)
    spatial_positions = normalized_rand_pos * level_shapes[:, None, :]

    return spatial_positions, batch_indices, level_indices, head_indices


@pytest.mark.cpu_and_cuda
class TestMultilevelSparseBilinearGridSample:
    @given(inputs=grid_sample_strategy())
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    def test_basics(self, inputs, device: Union[str, torch.device]):
        data = grid_sample_tensors(**inputs, device=device)
        assert data["sparse_tensor"] is not None
        assert data["spatial_positions"] is not None

        out = multilevel_sparse_bilinear_grid_sample(
            **data  #  pyright: ignore[reportArgumentType]
        )

        assert out is not None
        assert out.dtype == data["sparse_tensor"].dtype

        if data["level_indices"] is not None:
            L = data["level_indices"].shape[0]
        else:
            L = data["spatial_positions"].shape[-3]

        if data["head_indices"] is not None:
            H = data["head_indices"].shape[0]
        else:
            H = data["spatial_positions"].shape[-2]

        expected_shape = (
            sum(inputs["seq_lengths"]),
            *inputs["extra_batch_dims"],
            L,
            H,
            inputs["head_dim"],
        )
        assert out.shape == expected_shape

        if data["sparse_tensor"].requires_grad:
            out.sum().backward()
            assert data["sparse_tensor"].grad is not None
            if data["background_embedding"] is not None:
                assert data["background_embedding"].grad is not None

    @example(
        inputs={
            "n_heads": 1,
            "head_dim": 2,
            "embed_dim": 2,
            "n_levels": 1,
            "extra_batch_dims": [],
            "seq_lengths": [0],
            "level_spatial_shapes": np.array([[1, 1]]),
            "sparsity": 0.5,
            "require_grads": False,
            "make_level_indices": False,
            "make_head_indices": False,
            "make_background_embedding": False,
            "seed": 0,
        },
        n_pts=1,
    )
    @given(inputs=grid_sample_strategy(), n_pts=st.integers(0, 1000))
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    def test_pixel_centers(self, inputs, n_pts: int, device: Union[str, torch.device]):
        data = grid_sample_tensors(**inputs, device=device)
        assert data["sparse_tensor"] is not None
        assert data["level_spatial_shapes"] is not None
        sparse_tensor = data["sparse_tensor"]
        level_spatial_shapes = data["level_spatial_shapes"]

        spatial_positions, batch_indices, level_indices, head_indices = (
            make_random_sampling_points(
                sparse_tensor,
                level_spatial_shapes,
                n_pts,
                level_indices=data["level_indices"],
                head_indices=data["head_indices"],
            )
        )

        batch_size, _, _, n_levels, n_heads, head_dim = sparse_tensor.shape
        background_embedding = torch.randn(
            batch_size, n_levels, n_heads, head_dim, device=device
        )

        # enforce spatial positions at pixel centers
        spatial_positions = spatial_positions.floor() + 0.5

        out = multilevel_sparse_bilinear_grid_sample(
            sparse_tensor,
            spatial_positions,
            batch_indices,
            level_spatial_shapes,
            level_indices,
            head_indices,
            background_embedding,
        )

        if n_pts == 0:
            assert out.numel() == 0
            return

        # obtain expected pixel values from integer indexing
        i_idx, j_idx = spatial_positions.floor().long().unbind(-1)

        # broadcast batch, level, head indices to (n_pts, L, H)
        n_pts_, L, H = i_idx.shape
        batch_indices_exp = batch_indices.view(n_pts_, 1, 1).expand(n_pts_, L, H)
        level_indices_exp = level_indices.view(1, L, 1).expand_as(batch_indices_exp)
        head_indices_exp = head_indices.view(1, 1, H).expand_as(batch_indices_exp)

        int_indices = torch.stack(
            (batch_indices_exp, i_idx, j_idx, level_indices_exp, head_indices_exp), -1
        )
        int_indices_2d = int_indices.view(-1, 5)

        int_values_2d, is_specified = batch_sparse_index(sparse_tensor, int_indices_2d)

        # fill in background embedding where the sparse tensor has no specified value
        if background_embedding is not None:
            not_specified = ~is_specified
            bg_val = background_embedding[
                int_indices_2d[not_specified, 0],  # batch
                int_indices_2d[not_specified, 3],  # level
                int_indices_2d[not_specified, 4],  # head
            ]
            int_values_2d[not_specified] = bg_val

        expected = int_values_2d.view(n_pts_, L, H, -1)

        assert out.shape == expected.shape
        assert torch.allclose(out, expected)

    def test_2x2_grid_unit(self, device: Union[str, torch.device]):
        """Tests interpolation on a simple 2x2 image"""
        device = torch.device(device)
        tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
        # (batch_size, h, w, n_levels, n_heads, head_dim)
        tensor = tensor.view(1, 2, 2, 1, 1, 1)
        tensor = tensor.to_sparse(dense_dim=1)

        # sampling points: 0,0 pixel center and midpoint of grid
        spatial_positions = torch.tensor([[0.5, 0.5], [1.0, 1.0]], device=device)
        # (n_pts, n_levels, n_heads, 2)
        spatial_positions = spatial_positions[:, None, None, :]

        batch_indices = torch.tensor([0, 0], device=device)
        level_spatial_shapes = torch.tensor([[2, 2]], device=device)

        out = multilevel_sparse_bilinear_grid_sample(
            tensor,
            spatial_positions,
            batch_indices,
            level_spatial_shapes,
        ).squeeze()
        expected = out.new_tensor([0.0, 1.5], device=device)
        assert torch.allclose(out, expected)

    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(
        batch_size=st.integers(1, 3),
        height=st.integers(2, 32),
        width=st.integers(2, 32),
        head_dim=st.integers(1, 16),
        n_points_per_batch=st.integers(1, 32),
        seed=st.integers(0, 2**10),
    )
    def test_dense_grid_sample_hypothesis(
        self,
        batch_size: int,
        height: int,
        width: int,
        head_dim: int,
        n_points_per_batch: int,
        seed: int,
        device: Union[str, torch.device],
    ):
        """Test sparse bilinear sampling against an equivalent usage of F.grid_sample
        on a dense tensor.
        """
        device = torch.device(device)
        torch.manual_seed(seed)

        dense = torch.randn(batch_size, head_dim, height, width, device=device)
        # (batch_size, height, width, n_levels (=1), n_heads (=1), head_dim)
        sparse = sparse_split_heads(
            dense.permute(0, 2, 3, 1).unsqueeze(-2).to_sparse(dense_dim=1), 1
        )

        level_spatial_shapes = dense.new_tensor([[height, width]], dtype=torch.long)

        # make sampling points with a 0.5 margin to avoid edges
        margin = 0.5
        pos = torch.rand(batch_size, n_points_per_batch, 2, device=device)
        pos = pos * (level_spatial_shapes - 1 - 2 * margin) + margin

        # flatten across batches and add singleton level and head dims
        spatial_positions = pos.view(-1, 1, 1, 2)
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            n_points_per_batch,
        )

        # normalized positions within [-1, 1] for F.grid_sample
        # shape: batch_size, n_pts, 1, 2, with last dim [x, y] instead of [i, j]
        norm_pos = (pos / level_spatial_shapes * 2.0 - 1.0).unsqueeze(2).flip(-1)

        # do the interpolations
        out_sparse = multilevel_sparse_bilinear_grid_sample(
            sparse,
            spatial_positions,
            batch_indices,
            level_spatial_shapes,
        ).view(batch_size * n_points_per_batch, head_dim)

        out_dense = F.grid_sample(
            dense, norm_pos, mode="bilinear", align_corners=False
        ).squeeze(
            -1
        )  # (batch_size, head_dim, n_points_per_batch)
        out_dense = out_dense.permute(0, 2, 1).reshape(-1, head_dim)

        assert torch.allclose(out_sparse, out_dense, atol=1e-5, rtol=1e-4)


@pytest.mark.cpu_and_cuda
class TestMakeIndexAndWeightTensors:
    def test_basic(self, device: Union[str, torch.device]):
        device = torch.device(device)
        data = simple_sparse_input_tensors(device=device, random_seed=0)

        sparse_tensor = data["stacked_feature_maps"]
        level_spatial_shapes = data["level_spatial_shapes"]
        n_levels = level_spatial_shapes.size(0)

        n_heads = 4

        sparse_tensor_split = sparse_split_heads(sparse_tensor, n_heads=n_heads)

        # generate sampling points
        n_pts = 12
        torch.manual_seed(0)
        level_indices = torch.arange(n_levels, device=device)
        spatial_positions = torch.rand(
            (n_pts, n_levels, n_heads, 2), device=device
        ) * level_spatial_shapes[level_indices].unsqueeze(1)
        batch_indices = level_indices.new_zeros(n_pts)
        head_indices = torch.arange(n_heads, device=device)

        index_tensor, weight_tensor = _make_index_and_weight_tensors(
            spatial_positions,
            batch_indices,
            level_indices,
            level_spatial_shapes,
            head_indices,
        )

        indexed_values, _ = batch_sparse_index(sparse_tensor_split, index_tensor)
        assert indexed_values is not None
        assert isinstance(indexed_values, Tensor)

        out = (weight_tensor.unsqueeze(-2) @ indexed_values).squeeze(-2)
        assert out is not None
        assert out.shape == (n_pts, n_levels, n_heads, sparse_tensor_split.shape[-1])

    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(inputs=grid_sample_strategy())
    def test_weights_sum_to_one_hypothesis(
        self, inputs, device: Union[str, torch.device]
    ):
        device = torch.device(device)
        data = grid_sample_tensors(**inputs, device=device)

        assert data["spatial_positions"] is not None
        assert data["batch_indices"] is not None
        assert data["level_spatial_shapes"] is not None
        if data["level_indices"] is None:
            data["level_indices"] = torch.arange(inputs["n_levels"], device=device)
        if data["head_indices"] is None:
            data["head_indices"] = torch.arange(inputs["n_heads"], device=device)

        _, weights = _make_index_and_weight_tensors(
            data["spatial_positions"],
            data["batch_indices"],
            data["level_indices"],
            data["level_spatial_shapes"],
            data["head_indices"],
        )
        assert torch.allclose(weights.sum(-1), weights.new_ones([]))


@pytest.mark.cpu_and_cuda
class TestSparseSplitHeads:
    def test_basics(self, device: str):
        inputs = simple_sparse_input_tensors(device=device)

        sparse_tensor = inputs["stacked_feature_maps"]
        n_heads = 4

        split = sparse_split_heads(sparse_tensor, n_heads=4)

        assert split.shape == (
            *sparse_tensor.shape[:-1],
            n_heads,
            sparse_tensor.shape[-1] // n_heads,
        )

        # test some random indices to make sure the embeddings are correctly split
        for _ in range(5):
            i = random.randint(0, sparse_tensor._nnz() - 1)
            spatial_index = sparse_tensor.indices()[:, i]

            embedding = sparse_tensor[tuple(spatial_index)]

            split_embedding = []
            for h in range(n_heads):
                split_embedding.append(split[tuple(spatial_index) + (h,)])
            stacked_embedding = torch.cat(split_embedding)

            assert torch.equal(embedding, stacked_embedding)

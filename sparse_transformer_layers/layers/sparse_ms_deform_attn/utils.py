from typing import Optional

from torch import Tensor
import torch

from pytorch_sparse_utils.indexing import batch_sparse_index


@torch.jit.script
def sparse_split_heads(sparse_tensor: Tensor, n_heads: int) -> Tensor:
    """
    Splits a sparse tensor into multiple heads.

    Args:
        sparse_tensor (Tensor): The input sparse tensor.
        n_heads (int): The number of heads to split into.

    Returns:
        Tensor: The split sparse tensor with shape (*sparse_tensor.shape[:-1], n_heads, head_dim).
    """
    assert isinstance(sparse_tensor, Tensor)
    assert sparse_tensor.is_sparse
    n_specified_elements = sparse_tensor.indices().shape[1]
    embed_dim = sparse_tensor.shape[-1]
    assert embed_dim % n_heads == 0
    head_dim = embed_dim // n_heads
    repeated_indices = torch.repeat_interleave(sparse_tensor.indices(), n_heads, 1)
    new_indices = torch.cat(
        [
            repeated_indices,
            torch.arange(n_heads, device=sparse_tensor.device)
            .repeat(n_specified_elements)
            .unsqueeze(0),
        ]
    )
    new_values = sparse_tensor.values().view(n_specified_elements * n_heads, head_dim)

    new_shape = sparse_tensor.shape[:-1]
    new_shape = new_shape + (n_heads, head_dim)
    new_sparse_tensor = torch.sparse_coo_tensor(
        new_indices, new_values, new_shape, is_coalesced=sparse_tensor.is_coalesced()
    ).coalesce()
    return new_sparse_tensor


@torch.jit.script
def _make_index_and_weight_tensors(
    spatial_positions: Tensor,
    batch_indices: Tensor,
    level_indices: Tensor,
    level_spatial_shapes: Tensor,
    head_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Helper function to create the 4-point bilinear interpolation index tensor and
    the tensor of interpolation weights.
    """
    leading_batch_dims = spatial_positions.shape[:-1]

    index_tensor = spatial_positions.new_empty(
        leading_batch_dims + (4, 5), dtype=torch.long
    )

    # broadcast batch indices over level, head, interpolant dims
    index_tensor[..., 0] = batch_indices[..., None, None, None]

    spatial_positions_exp = spatial_positions.unsqueeze(-2)  # add interpolant dim
    # convert from pixel coordinates to pixel indices
    index_tensor[..., 1:3] = spatial_positions_exp.floor().long()
    # if the sampling point is above/left of the pixel center, then offset by -1
    index_tensor[..., 1:3] -= torch.ones_like(index_tensor[..., 1:3]) * (
        spatial_positions_exp.frac() < 0.5
    )

    # 1-pixel shifts for the other 3 interpolant points
    pos_offset = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=index_tensor.dtype,
        device=index_tensor.device,
    )
    index_tensor[..., 1:3] += pos_offset

    # distance from nearest 4 pixel centers
    delta = (spatial_positions_exp - (index_tensor[..., 1:3] + 0.5)).abs()
    weights = (1.0 - delta).prod(-1)
    assert torch.allclose(weights.sum(-1), weights.new_ones([]))

    level_shapes_broadcasted = level_spatial_shapes[level_indices][:, None, None, :]
    index_tensor[..., 1:3].clamp_(index_tensor.new_zeros([]), level_shapes_broadcasted)

    index_tensor[..., -2] = level_indices[:, None, None]

    index_tensor[..., -1] = head_indices[:, None]

    return index_tensor, weights


@torch.jit.script
def multilevel_sparse_bilinear_grid_sample(
    sparse_tensor: Tensor,
    spatial_positions: Tensor,
    batch_indices: Tensor,
    level_spatial_shapes: Tensor,
    level_indices: Optional[Tensor] = None,
    head_indices: Optional[Tensor] = None,
    background_embedding: Optional[Tensor] = None,
) -> Tensor:
    """Bilinearly samples into a 2D sparse tensor. Similar to F.grid_sample with
    align_corners=False except the sampled tensor is expected to be sparse, the points
    are not in a grid, and the sampling points are expected to be in absolute
    coordinates instead of normalized to [-1, -1].

    Note that this function uses a coordinate system that places coordinate (i, j) at
    the upper left corner of pixel [i, j]. The center of the pixel [i, j] is thus at
    coordinate (i+0.5, j+0.5). Interpolation is done from the 4 closest pixel centers
    to each sampling point in `spatial_positions`.

    The batch size (number of images), number of feature levels, and number of attention
    heads are inferred from the shape of `sparse_tensor`.

    Args:
        sparse_tensor (Tensor): torch.sparse.sparse_coo_tensor with shape
            (batch_size, height, width, n_levels, n_heads, head_dim), with the last
            dimension being dense and other dimensions sparse.
        spatial_positions (Tensor): Sampling point coordinates, with shape
            (N, ..., L, H, 2), with the last dimension in order (i, j),
            as in sparse_tensor, N being a batch dimension, L and H being level and head
            dimensions, respectively, and ... being additional optional batch dimensions.
        batch_indices (Tensor): Tensor of shape (N, ...) with the index of the batch
            element (image index) for each point in spatial_positions
        level_spatial_shapes (Tensor): (n_levels, 2) tensor with the (height, width) of
            each level's feature map
        level_indices (Optional[Tensor]): Tensor of shape (L) with the index of the
            level for each point in spatial_positions. If None, it defaults to
            torch.arange(n_levels).
        head_indices (Optional[Tensor]): Tensor of shape (H) with the index of the head
            to be sampled for each point in spatial_positions. If None, it defaults to
            torch.arange(n_heads).
        background_embedding (Optional[Tensor]): Tensor of shape
            (batch, n_levels, n_heads, head_dim) that should be used as an interpolant
            for points that are not specified in sparse_tensor. If not given, a 0
            vector will be used instead.

    Returns:
        Tensor: Bilinear interpolated tensor of shape (N, ..., L, H, head_dim).
    """
    assert sparse_tensor.is_sparse
    if sparse_tensor.ndim != 6:
        raise ValueError(
            "Expected 6D sparse tensor (b, h, w, l, h, d), got shape "
            f"{sparse_tensor.shape}"
        )
    if sparse_tensor.sparse_dim() != 5:
        raise ValueError(
            "Expected sparse tensor to have 5 sparse dims, got "
            f"{sparse_tensor.sparse_dim()}"
        )
    if spatial_positions.shape[:-3] != batch_indices.shape:
        raise ValueError(
            "Shape mismatch for spatial_positions and batch_indices: got "
            f"{spatial_positions.shape} and {batch_indices.shape}"
        )
    if level_indices is not None and spatial_positions.size(-3) != level_indices.size(
        0
    ):
        raise ValueError(
            "Shape mismatch for spatial_positions and level_indices: got "
            f"{spatial_positions.shape} and {level_indices.shape}"
        )
    if head_indices is not None and spatial_positions.size(-2) != head_indices.size(0):
        raise ValueError(
            "Shape mismatch for spatial_positions and head_indices: got "
            f"{spatial_positions.shape} and {head_indices}.shape"
        )
    if sparse_tensor.size(-3) != level_spatial_shapes.size(0):
        raise ValueError(
            "Number of feature levels in level_spatial_shapes does not match number of "
            "levels in sparse_tensor: "
            f"{level_spatial_shapes.size(0)} != {sparse_tensor.size(-3)}."
        )

    # Default level and head indices
    if level_indices is None:
        n_levels = sparse_tensor.shape[-3]
        if n_levels != spatial_positions.shape[-3]:
            raise RuntimeError(
                "Default level_indices mode assumes that spatial_positions.shape[-3] "
                f"== n_levels, but got {spatial_positions.shape[-3] != n_levels}."
            )
        level_indices = torch.arange(n_levels, device=spatial_positions.device)
    if head_indices is None:
        n_heads = sparse_tensor.shape[-2]
        if n_heads != spatial_positions.shape[-2]:
            raise RuntimeError(
                "Default head_indices mode assumes that spatial_positions.shape[-2] "
                f"== n_heads, but got {spatial_positions.shape[-2]} and {n_heads}."
            )
        head_indices = torch.arange(n_heads, device=spatial_positions.device)

    index_tensor, weights = _make_index_and_weight_tensors(
        spatial_positions,
        batch_indices,
        level_indices,
        level_spatial_shapes,
        head_indices,
    )

    val, is_specified = batch_sparse_index(sparse_tensor, index_tensor)
    if background_embedding is not None:
        assert (
            background_embedding.shape
            == (sparse_tensor.size(0),) + sparse_tensor.shape[-3:]
        )
        # add batch dims and 4-point interpolation dim and broadcast
        background_embedding = background_embedding.unsqueeze(-2)[
            batch_indices
        ].expand_as(val)
        mask = is_specified.unsqueeze(-1).expand_as(val).logical_not()
        val.masked_scatter_(mask, background_embedding[mask])

    out = torch.matmul(weights.unsqueeze(-2).to(val), val).squeeze(-2)

    return out

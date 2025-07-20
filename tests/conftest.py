import pytest
import torch

from .constants import (
    BATCH_SIZE,
    SPARSE_DIM_1,
    SPARSE_DIM_2,
    SPARSE_DIM_3,
    EMBED_DIM,
    ALWAYS_SPECIFIED,
    ALWAYS_UNSPECIFIED,
)


@pytest.fixture
def setup_sparse_tensor(device):
    """Create a 4D sparse tensor with vectorized operations."""
    # Create coordinates spanning all dimensions
    grid = torch.meshgrid(
        torch.arange(BATCH_SIZE, device=device),
        torch.arange(SPARSE_DIM_1, device=device),
        torch.arange(SPARSE_DIM_2, device=device),
        torch.arange(SPARSE_DIM_3, device=device),
        indexing="ij",
    )
    coords = torch.stack(grid, dim=-1).reshape(-1, 4)

    # Filter out unspecified points with vectorized operations
    mask_per_unspecified = (
        coords.unsqueeze(1) == ALWAYS_UNSPECIFIED.to(device).unsqueeze(0)
    ).all(dim=2)
    match_any_unspecified = mask_per_unspecified.any(dim=1)
    valid_coords = coords[~match_any_unspecified]

    # Sample a random subset
    num_samples = max(50, int(valid_coords.shape[0] * 0.15))  # Fixed 15% sample rate
    indices = torch.randperm(valid_coords.shape[0])[:num_samples]
    sampled_coords = valid_coords[indices]

    # Add specified test points
    final_coords = torch.cat([sampled_coords, ALWAYS_SPECIFIED.to(device)], dim=0)

    # Create sparse tensor
    values = torch.randn(
        final_coords.shape[0],
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )
    return torch.sparse_coo_tensor(
        final_coords.t(),
        values,
        (BATCH_SIZE, SPARSE_DIM_1, SPARSE_DIM_2, SPARSE_DIM_3, EMBED_DIM),
    ).coalesce()

import pytest
import torch

from ..constants import (
    BATCH_SIZE,
    SPARSE_DIM_1,
    SPARSE_DIM_2,
    SPARSE_DIM_3,
    ALWAYS_SPECIFIED,
    ALWAYS_UNSPECIFIED,
)


@pytest.fixture
def setup_linear_index_tensor(setup_sparse_tensor, device):
    """Create index tensor for linear mapping tests with each index having a 50% chance
    of being from specified indices and 50% chance of being random."""
    # Get indices from the sparse tensor
    sparse_indices = setup_sparse_tensor.indices().t().contiguous()

    # Assert that we have enough sparse indices for meaningful testing
    assert len(sparse_indices) > 0, "Sparse tensor has no indices for testing"

    # Determine total number of indices to generate (excluding test points)
    total_indices = 50
    num_test_points = len(ALWAYS_SPECIFIED) + len(ALWAYS_UNSPECIFIED)
    num_random_indices = total_indices - num_test_points

    # Create a batch-to-indices mapping for quick lookup
    batch_to_sparse_indices = {
        b: sparse_indices[sparse_indices[:, 0] == b] for b in range(BATCH_SIZE)
    }

    # Generate indices with 50% probability of being specified vs random
    random_indices = torch.zeros(num_random_indices, 4, dtype=torch.long, device=device)

    # First assign random batch indices
    random_indices[:, 0].random_(0, BATCH_SIZE)

    # For each index, decide whether to use specified or random values
    use_specified = torch.rand(num_random_indices, device=device) < 0.5

    # Default all spatial dimensions to random values
    random_indices[:, 1].random_(0, SPARSE_DIM_1)
    random_indices[:, 2].random_(0, SPARSE_DIM_2)
    random_indices[:, 3].random_(0, SPARSE_DIM_3)

    # Vectorized replacement of specified indices
    # Process each batch group separately
    for batch_idx in range(BATCH_SIZE):
        # Find indices that should use specified values from this batch
        batch_mask = (random_indices[:, 0] == batch_idx) & use_specified
        num_to_replace = int(batch_mask.sum().item())

        if num_to_replace > 0 and batch_idx in batch_to_sparse_indices:
            batch_specified = batch_to_sparse_indices[batch_idx]
            if len(batch_specified) > 0:
                # Sample indices with replacement
                sampled_idx = torch.randint(0, len(batch_specified), (num_to_replace,))
                sampled_specified = batch_specified[sampled_idx]

                # Update all matching indices at once
                random_indices[batch_mask, 1:] = sampled_specified[:, 1:]

    # Combine with test points
    combined_indices = torch.cat(
        [
            random_indices,
            ALWAYS_SPECIFIED.to(device),  # Test points known to be in the sparse tensor
            ALWAYS_UNSPECIFIED.to(device),  # Test points known to be unspecified
        ],
        dim=0,
    )

    # Sort by batch dimension for consistent batch-wise processing
    _, sort_indices = torch.sort(combined_indices[:, 0])
    return combined_indices[sort_indices].contiguous()

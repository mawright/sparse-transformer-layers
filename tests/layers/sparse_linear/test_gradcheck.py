import pytest
import torch
from torch.autograd import gradcheck

from pytorch_sparse_utils.indexing.utils import get_sparse_index_mapping

# Import module functions
from sparse_transformer_layers.layers.sparse_linear.autograd import (
    GatherAndLinearFunction,
)

from ..constants import EMBED_DIM


@pytest.mark.cuda_if_available
def test_gather_and_linear_function(
    setup_sparse_tensor, setup_linear_index_tensor, device
):
    """Test gradient computation for gather and linear function."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_linear_index_tensor

    # Get index mapping
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    # Initialize parameters
    weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    bias = torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)

    # Run gradcheck
    inputs = (sparse_tensor_values, index_search, is_specified_mask, weight, bias)
    assert gradcheck(
        GatherAndLinearFunction.apply, inputs  # pyright: ignore[reportArgumentType]˝
    )

import pytest
import torch

from sparse_transformer_layers.sparse_linear import batch_sparse_index_linear

from ..constants import EMBED_DIM

@pytest.mark.cuda_if_available
@pytest.mark.parametrize(
    "include_bias", [True, False], ids=["include_bias=True", "include_bias=False"]
)
def test_end_to_end_gather_linear(
    setup_sparse_tensor, setup_linear_index_tensor, include_bias, device
):
    """Test end-to-end gather and linear mapping."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_linear_index_tensor

    # Initialize parameters
    weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    bias = (
        torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)
        if include_bias
        else None
    )

    # Run the operation
    transformed, is_specified_mask = batch_sparse_index_linear(
        sparse_tensor, index_tensor, weight, bias
    )

    # Check output shape
    assert transformed.shape == (index_tensor.shape[0], EMBED_DIM)
    assert is_specified_mask.shape == (index_tensor.shape[0],)

    # Compute loss and check gradient flow
    loss = transformed.sum()
    loss.backward()

    assert weight.grad is not None
    if include_bias:
        assert bias is not None
        assert bias.grad is not None

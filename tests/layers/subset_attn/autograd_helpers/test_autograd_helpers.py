import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import Tensor

from sparse_transformer_layers.layers.subset_attn.autograd_helpers import (
    permute_for_attention,
    permute_for_attention_backward,
    split_heads,
)


@pytest.mark.cpu_and_cuda
class TestSplitHeads:

    # Basic functionality tests

    def test_split_heads_2d(self):
        """Test splitting a 2D tensor."""
        tensor = torch.randn(5, 12)  # 5 queries, 12-dimensional embedding
        n_heads = 3
        result = split_heads(tensor, n_heads)

        # Check shape
        assert result.shape == (5, 3, 4)  # [n_queries, n_heads, embed_dim // n_heads]

        # Check that the content is preserved
        assert torch.allclose(result.reshape(5, 12), tensor)

    def test_split_heads_3d(self):
        """Test splitting a 3D tensor."""
        tensor = torch.randn(5, 7, 12)  # 5 queries, 7 keys per query, 12-dim embedding
        n_heads = 4
        result = split_heads(tensor, n_heads)

        # Check shape
        assert result.shape == (
            5,
            7,
            4,
            3,
        )  # [n_queries, n_keys_per_query, n_heads, embed_dim // n_heads]

        # Check that the content is preserved
        assert torch.allclose(result.reshape(5, 7, 12), tensor)

    # Edge cases

    def test_split_heads_exact_division(self):
        """Test when the last dimension is exactly equal to n_heads."""
        tensor = torch.randn(5, 8)
        n_heads = 8
        result = split_heads(tensor, n_heads)

        # Each head should have dimension 1
        assert result.shape == (5, 8, 1)

    def test_split_heads_single_head(self):
        """Test with n_heads = 1."""
        tensor = torch.randn(5, 12)
        n_heads = 1
        result = split_heads(tensor, n_heads)

        # Should just add a dimension
        assert result.shape == (5, 1, 12)
        assert torch.allclose(result.squeeze(1), tensor)

    def test_split_heads_empty_valid(self):
        """Test with empty but valid tensors."""
        tensor = torch.randn(0, 12)  # No queries
        n_heads = 3
        result = split_heads(tensor, n_heads)

        assert result.shape == (0, 3, 4)

    # Error cases

    def test_split_heads_indivisible(self):
        """Test error when the last dimension is not divisible by n_heads."""
        tensor = torch.randn(5, 10)  # 10 is not divisible by 3
        n_heads = 3

        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Last dimension of tensor",
        ):
            split_heads(tensor, n_heads)

    def test_split_heads_invalid_dim(self):
        """Test error with tensors of invalid dimensions."""
        # 1D tensor
        tensor_1d = torch.randn(10)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 2D or 3D",
        ):
            split_heads(tensor_1d, 2)

        # 4D tensor
        tensor_4d = torch.randn(5, 7, 9, 12)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 2D or 3D",
        ):
            split_heads(tensor_4d, 3)

    # Property-based tests

    @given(
        n_queries=st.integers(1, 10),
        head_dim=st.integers(1, 10),
        n_heads=st.integers(1, 10),
    )
    def test_property_2d_preserves_elements(self, n_queries, head_dim, n_heads):
        """Test that the total number of elements is preserved for 2D tensors."""
        # Ensure embed_dim is divisible by n_heads
        embed_dim = head_dim * n_heads

        tensor = torch.randn(n_queries, embed_dim)
        result = split_heads(tensor, n_heads)

        assert tensor.numel() == result.numel()
        assert torch.allclose(result.reshape(tensor.shape), tensor)

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        head_dim=st.integers(1, 10),
        n_heads=st.integers(1, 10),
    )
    def test_property_3d_preserves_elements(self, n_queries, n_keys, head_dim, n_heads):
        """Test that the total number of elements is preserved for 3D tensors."""
        # Ensure embed_dim is divisible by n_heads
        embed_dim = head_dim * n_heads

        tensor = torch.randn(n_queries, n_keys, embed_dim)
        result = split_heads(tensor, n_heads)

        assert tensor.numel() == result.numel()
        assert torch.allclose(result.reshape(tensor.shape), tensor)

    @given(
        n_queries=st.integers(1, 10),
        head_dim=st.integers(1, 10),
        n_heads=st.integers(1, 10),
    )
    def test_property_consistent_with_manual(self, n_queries, head_dim, n_heads):
        """Test that the function is consistent with manual reshaping."""
        # Ensure embed_dim is divisible by n_heads
        embed_dim = head_dim * n_heads

        tensor = torch.randn(n_queries, embed_dim)
        result = split_heads(tensor, n_heads)

        # Manually reshape
        manual_result = tensor.reshape(n_queries, n_heads, embed_dim // n_heads)

        assert torch.allclose(result, manual_result)


@pytest.mark.cpu_and_cuda
class TestPermuteForAttention:
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
        deadline=None,
    )
    @given(dims=st.lists(st.integers(0, 8), min_size=3, max_size=4))
    def test_hypothesis(self, dims: list[int], device):
        tensor = torch.randn(dims, device=device)
        permuted: Tensor = permute_for_attention(tensor)

        assert isinstance(permuted, Tensor)
        assert permuted.is_contiguous()
        if tensor.ndim == 3:
            assert permuted.shape == (tensor.size(1), tensor.size(0), tensor.size(2))
        else:
            assert permuted.shape == (
                tensor.size(2),
                tensor.size(0),
                tensor.size(1),
                tensor.size(3),
            )

    @pytest.mark.parametrize("shape", [[2, 4], [2, 4, 6, 8, 10]])
    def test_shape_error(self, shape: list[int], device):
        tensor = torch.randn(4, 6, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 3D or 4D",
        ):
            _ = permute_for_attention(tensor)


@pytest.mark.cpu_and_cuda
class TestPermuteForAttentionBackward:
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
        deadline=None,
    )
    @given(dims=st.lists(st.integers(0, 8), min_size=3, max_size=4))
    def test_hypothesis(self, dims: list[int], device):
        tensor = torch.randn(dims, device=device)
        permuted: Tensor = permute_for_attention_backward(tensor)

        assert isinstance(permuted, Tensor)
        assert permuted.is_contiguous()
        if tensor.ndim == 3:
            assert permuted.shape == (tensor.size(1), tensor.size(0), tensor.size(2))
        else:
            assert permuted.shape == (
                tensor.size(1),
                tensor.size(2),
                tensor.size(0),
                tensor.size(3),
            )

    @pytest.mark.parametrize("shape", [[2, 4], [2, 4, 6, 8, 10]])
    def test_shape_error(self, shape: list[int], device):
        tensor = torch.randn(4, 6, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 3D or 4D",
        ):
            _ = permute_for_attention_backward(tensor)

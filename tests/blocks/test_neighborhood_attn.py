import math
from typing import Any, Optional, Union

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from nd_rotary_encodings import (
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
)
from pytorch_sparse_utils.batching import batch_offsets_to_indices
from torch import Tensor, nn

from sparse_transformer_layers.blocks.neighborhood_attn import (
    SparseNeighborhoodAttentionBlock,
    get_multilevel_neighborhoods,
)
from sparse_transformer_layers.layers.subset_attn import (
    BatchSparseIndexSubsetAttention,
)
from ..conftest import random_multilevel_sparse_tensor_indices, simple_sparse_input_tensors
from .conftest import ModuleHook


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for SparseNeighborhoodAttentionBlock tests."""
    return {
        "embed_dim": 64,
        "n_heads": 4,
        "n_levels": 3,
        "neighborhood_sizes": [3, 5, 7],
        "position_dim": 2,
        "dropout": 0.0,
        "bias": False,
        "norm_first": True,
        "rope_spatial_base_theta": 100.0,
        "rope_level_base_theta": 10.0,
        "rope_share_heads": False,
        "rope_freq_group_pattern": "single",
    }


@pytest.fixture
def base_module_instance(
    base_config: dict[str, Any], device: str
) -> SparseNeighborhoodAttentionBlock:
    """Create a module instance for testing."""
    return SparseNeighborhoodAttentionBlock(**base_config).to(device)


@st.composite
def neighborhood_data_strategy(
    draw,
    require_grads: bool = False,
    standard_float_range=False,
) -> dict[str, Any]:
    # Draw basic shape parameters
    n_heads = draw(st.integers(1, 8))
    head_dim = draw(st.integers(1, 16)) * 2
    embed_dim = n_heads * head_dim
    position_dim = draw(st.integers(1, 3))
    n_levels = draw(st.integers(1, 4))

    # Draw data size parameters
    seq_lengths = draw(st.lists(st.integers(0, 32), min_size=1, max_size=4))
    batch_size = len(seq_lengths)
    query_batch_offsets = np.zeros((batch_size + 1,), dtype=int)
    query_batch_offsets[1:] = np.cumsum(seq_lengths)

    # Draw neighborhood sizes
    neighborhood_sizes = draw(
        st.lists(
            st.integers(1, 7).map(lambda x: 2 * x + 1),  # Ensure odd values
            min_size=n_levels,
            max_size=n_levels,
        )
    )

    # Level shapes
    level_shapes = []
    last_level = [1] * position_dim
    for level in range(n_levels):
        shape = []
        for pos_dim in range(position_dim):
            shape.append(draw(st.integers(last_level[pos_dim], 1000 * (level + 1))))
        last_level = shape
        level_shapes.append(shape)

    level_shapes = np.array(level_shapes)
    assert np.array_equal(np.sort(level_shapes, 0), level_shapes)

    # Draw tensor generation parameters
    seed = draw(st.integers(0, int(1e8)))
    float_dtype = draw(st.just(torch.float32))

    if standard_float_range:
        min_float_value = -1
        max_float_value = 1
    else:
        min_float_value = draw(
            st.floats(min_value=-1e6, max_value=1e6, exclude_max=True)
        )
        max_float_value = draw(st.floats(min_value=min_float_value, max_value=1e6))

    position_dtype = draw(st.just(torch.float32))

    if require_grads:
        tensors_requiring_grads = draw(
            st.lists(
                st.sampled_from(
                    ["query", "query_spatial_positions", "stacked_feature_maps"]
                ),
                min_size=1,
                max_size=3,
                unique=True,
            )
        )
    else:
        tensors_requiring_grads = []

    # Portion of generated query positions that will have full neighborhoods
    query_full_neighborhood_portion = draw(st.floats(min_value=0.0, max_value=1.0))

    # Sparsity of the non-full-neighborhood regions
    sparse_region_sparsity = draw(st.floats(0.4, 1.0, exclude_max=True))

    # Select rope freq group pattern and determine if nonequal freq groups need to
    # be allowed
    rope_freq_group_pattern = draw(st.sampled_from(["single", "partition", "closure"]))
    if rope_freq_group_pattern == "partition":
        rope_enforce_freq_groups_equal = head_dim % (2 * 2) == 0
    elif rope_freq_group_pattern == "closure":
        rope_enforce_freq_groups_equal = head_dim % (2 * 3) == 0
    else:
        rope_enforce_freq_groups_equal = True

    # Decide whether to include optional level indices
    make_level_indices = draw(st.booleans())

    return {
        "config": {
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "n_levels": n_levels,
            "neighborhood_sizes": neighborhood_sizes,
            "position_dim": position_dim,
            "dropout": draw(st.floats(0.0, 1.0, exclude_max=True)),
            "bias": draw(st.booleans()),
            "norm_first": draw(st.booleans()),
            "rope_spatial_base_theta": draw(st.floats(1.0, 1000.0)),
            "rope_level_base_theta": draw(st.floats(1.0, 100.0)),
            "rope_share_heads": draw(st.booleans()),
            "rope_freq_group_pattern": rope_freq_group_pattern,
            "rope_enforce_freq_groups_equal": rope_enforce_freq_groups_equal,
        },
        "tensor_config": {
            "embed_dim": embed_dim,
            "position_dim": position_dim,
            "query_batch_offsets": query_batch_offsets,
            "level_spatial_shapes": level_shapes,
            "neighborhood_sizes": neighborhood_sizes,
            "query_full_neighborhood_portion": query_full_neighborhood_portion,
            "sparse_region_sparsity": sparse_region_sparsity,
            "make_level_indices": make_level_indices,
            "min_float_value": min_float_value,
            "max_float_value": max_float_value,
            "float_dtype": float_dtype,
            "position_dtype": position_dtype,
            "tensors_requiring_grads": tensors_requiring_grads,
            "seed": seed,
        },
    }


def strategy_input_tensors(
    embed_dim: int,
    position_dim: int,
    query_batch_offsets: Union[list[int], Tensor],
    level_spatial_shapes: Union[np.ndarray, Tensor],
    neighborhood_sizes: Union[list[int], Tensor],
    query_full_neighborhood_portion: float,
    sparse_region_sparsity: float,
    make_level_indices: bool,
    min_float_value: float,
    max_float_value: float,
    float_dtype: torch.dtype,
    position_dtype: torch.dtype,
    tensors_requiring_grads: list[str],
    seed: int,
    device: Union[str, torch.device],
) -> dict[str, Optional[torch.Tensor]]:
    """Create input tensors based on strategy parameters."""
    if isinstance(device, str):
        device = torch.device(device, 0)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    # convert non-tensors to tensors
    query_batch_offsets = torch.as_tensor(query_batch_offsets, device=device)
    level_spatial_shapes = torch.as_tensor(level_spatial_shapes, device=device)
    neighborhood_sizes = torch.as_tensor(neighborhood_sizes, device=device)

    n_queries = int(query_batch_offsets[-1].item())
    batch_size = len(query_batch_offsets) - 1
    n_levels = level_spatial_shapes.size(-2)

    # Create queries: embeddings and spatial positions
    query_batch_indices = batch_offsets_to_indices(query_batch_offsets)

    query = torch.empty((n_queries, embed_dim), device=device, dtype=float_dtype)
    query.uniform_(min_float_value, max_float_value)
    if "query" in tensors_requiring_grads:
        query.requires_grad_(True)

    max_spatial_shape = level_spatial_shapes.max(-2)[0]
    # if different spatial shapes for each batch, expand per batch
    if max_spatial_shape.ndim == 2:
        max_spatial_shape = max_spatial_shape[query_batch_indices]

    # make query positions in the region [0, max] for each dim
    query_spatial_positions = torch.empty(
        (n_queries, position_dim), device=device, dtype=position_dtype
    ).uniform_(0, 1)
    query_spatial_positions *= max_spatial_shape
    if "query_spatial_positions" in tensors_requiring_grads:
        query_spatial_positions.requires_grad_(True)

    # make level indices, or don't
    if make_level_indices:
        query_level_indices = torch.randint(
            level_spatial_shapes.size(-2), size=(n_queries,), device=device
        )
    else:
        # Every query is at maximum spatial shape (i.e. decoder object queries)
        query_level_indices = None

    # Create stacked feature maps sparse tensor
    # 1. Obtain full neighborhood indices for specified queries
    n_full_nhood_queries = int(n_queries * query_full_neighborhood_portion)
    shuffled_query_indices = torch.randperm(n_queries, device=device)

    full_neighborhood_query_idx = shuffled_query_indices[:n_full_nhood_queries]

    full_nhood_batch_idx = query_batch_indices[full_neighborhood_query_idx]
    full_nhood_positions = query_spatial_positions[full_neighborhood_query_idx]

    multilevel_nhoods, oob_mask, nhood_level_indices = get_multilevel_neighborhoods(
        full_nhood_positions, level_spatial_shapes, neighborhood_sizes
    )
    n_nhood_indices = multilevel_nhoods.size(1)
    nhood_indices = torch.cat(
        [
            full_nhood_batch_idx.view(-1, 1, 1).expand(-1, n_nhood_indices, 1),
            multilevel_nhoods,
            nhood_level_indices.view(1, -1, 1).expand(n_full_nhood_queries, -1, 1),
        ],
        dim=-1,
    )

    # 2. Randomly sample a portion of all indices to be nonzero
    sparse_nonzero_indices = random_multilevel_sparse_tensor_indices(
        level_spatial_shapes, sparse_region_sparsity, batch_size, 1000, device
    )

    # 3. Construct the sparse tensor
    all_nonzero_indices = torch.cat(
        [nhood_indices[~oob_mask].T, sparse_nonzero_indices], dim=1
    )  # (D x nnz)
    if max_spatial_shape.ndim == 2:
        max_spatial_shape = max_spatial_shape.max(0)[0]
    sparse_shape = [batch_size] + max_spatial_shape.tolist() + [n_levels, embed_dim]
    stacked_feature_maps = torch.sparse_coo_tensor(
        all_nonzero_indices,
        torch.randn(
            all_nonzero_indices.size(1), embed_dim, device=device, dtype=float_dtype
        ),
        sparse_shape,
        device=device,
    ).coalesce()

    # Check validity of the sparse tensor
    assert (stacked_feature_maps.indices() >= 0).all()
    for level in range(n_levels):
        level_mask = stacked_feature_maps.indices()[-1] == level
        level_indices = stacked_feature_maps.indices().T[level_mask, 1:-1]
        assert torch.all(level_indices < level_spatial_shapes[level])

    if "stacked_feature_maps" in tensors_requiring_grads:
        stacked_feature_maps.requires_grad_(True)

    # reset rng state
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state, device)
    else:
        torch.set_rng_state(rng_state)

    return {
        "query": query,
        "query_spatial_positions": query_spatial_positions,
        "query_batch_offsets": query_batch_offsets,
        "stacked_feature_maps": stacked_feature_maps,
        "level_spatial_shapes": level_spatial_shapes,
        "query_level_indices": query_level_indices,
    }


@pytest.mark.cuda_if_available
class TestGetMultilevelNeighborhoods:
    """Tests for get_multilevel_neighborhoods function."""

    def test_basic_functionality(self, device: str):
        """Test the basic functionality of get_multilevel_neighborhoods."""
        # Create 2D case with 2 query positions
        query_positions = torch.tensor(
            [
                [10.1, 20.2],  # Query position 1
                [15.3, 25.5],  # Query position 2
            ],
            device=device,
        )

        # Define 2 resolution levels
        level_shapes = torch.tensor(
            [
                [32, 32],  # Level 0 shape
                [16, 16],  # Level 1 shape
            ],
            device=device,
        )

        # Use simple neighborhood sizes
        neighborhood_sizes = [3, 5]

        # Call the function
        multilevel_neighborhood_indices, out_of_bounds_mask, level_indices = (
            get_multilevel_neighborhoods(
                query_positions, level_shapes, neighborhood_sizes
            )
        )

        # Validate output shapes
        n_queries, position_dim = query_positions.shape
        expected_total_elements = 3**2 + 5**2  # 9 + 25 = 34 elements
        assert multilevel_neighborhood_indices.shape == (
            n_queries,
            expected_total_elements,
            position_dim,
        )
        assert level_indices.shape == (expected_total_elements,)

        # Check level indices are correct
        assert torch.all(level_indices[:9] == 0)  # First 9 elements from level 0
        assert torch.all(level_indices[9:] == 1)  # Remaining from level 1

        # Check no out of bounds indices
        assert not out_of_bounds_mask.any()

    def test_input_validation(self, device: str):
        """Test error handling and input validation."""
        # Invalid query positions (should be 2D)
        invalid_query = torch.ones(3, 2, 2, device=device)  # 3D tensor
        valid_shapes = torch.tensor([[32, 32], [16, 16]], device=device)

        with pytest.raises((ValueError, torch.jit.Error)):  # type: ignore
            get_multilevel_neighborhoods(invalid_query, valid_shapes)

        # Invalid neighborhood sizes (should be odd)
        valid_query = torch.ones(3, 2, device=device)
        even_sizes = [2, 4]  # Even sizes should raise error

        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="odd neighborhood_sizes",
        ):
            get_multilevel_neighborhoods(valid_query, valid_shapes, even_sizes)

    def test_single_level(self, device: str):
        """Test with a single level for simplicity."""
        # 1D case with 1 query position
        query_positions = torch.tensor([[5.5]], device=device)

        # Single resolution level
        level_shapes = torch.tensor([[10]], device=device)

        # Use neighborhood size of 3
        neighborhood_sizes = [3]

        # Expected outputs for this simple case:
        # Query at position 5.5 gets floored to 5
        # With neighborhood size 3, we expect indices [4, 5, 6]
        expected_indices = torch.tensor([[[4], [5], [6]]], device=device)
        expected_level_indices = torch.tensor([0, 0, 0], device=device)

        # Call the function
        multilevel_neighborhood_indices, oob_mask, level_indices = (
            get_multilevel_neighborhoods(
                query_positions, level_shapes, neighborhood_sizes
            )
        )

        # Validate outputs match expected values
        assert torch.all(multilevel_neighborhood_indices == expected_indices)
        assert torch.all(level_indices == expected_level_indices)

        assert not oob_mask.any()

    def test_out_of_bounds_indices(self, device: str):
        """Test that out of bounds indices are properly masked out"""
        level_shapes = torch.tensor(
            [
                [5, 5],
                [10, 10],
            ],
            device=device,
        )

        # both levels same size to keep mask construction simple
        neighborhood_sizes = torch.tensor([3, 3], device=device)

        query_positions = torch.tensor(
            [
                [0.5, 0.5],  # upper left corner of neighborhood out of bounds
                [1.0, 2.5],  # out of bounds on top for smaller level
                [2.5, 2.5],  # in bounds
                [10.5, 10.5],  # out of bounds on all but upper left on both levels
            ],
            device=device,
        )

        n_queries = query_positions.size(0)
        n_levels = level_shapes.size(0)

        expected_out_of_bounds_mask = torch.zeros(
            n_queries,
            n_levels,
            3,
            3,
            device=device,
            dtype=torch.bool,
        )

        # first query
        expected_out_of_bounds_mask[0, :, 0] = True  # top part
        expected_out_of_bounds_mask[0, :, :, 0] = True  # left part
        # second query
        expected_out_of_bounds_mask[1, 0, 0] = True  # top part of coarser level
        # fourth query
        expected_out_of_bounds_mask[3, :, -2:] = True  # bottom part
        expected_out_of_bounds_mask[3, :, :, -2:] = True  # right part

        expected_out_of_bounds_mask = expected_out_of_bounds_mask.view(n_queries, -1)

        # Call the function
        _, out_of_bounds_mask, _ = get_multilevel_neighborhoods(
            query_positions, level_shapes, neighborhood_sizes
        )

        assert torch.equal(out_of_bounds_mask, expected_out_of_bounds_mask)

    def test_multiple_levels_3d(self, device: str):
        """Test 3D positions with multiple levels."""
        # 3D case
        query_positions = torch.tensor(
            [
                [5.5, 6.5, 7.5],  # Query position 1
                [8.2, 9.3, 10.4],  # Query position 2
            ],
            device=device,
        )

        # Define 2 resolution levels in 3D
        level_shapes = torch.tensor(
            [
                [12, 12, 12],  # Level 0 shape
                [6, 6, 6],  # Level 1 shape
            ],
            device=device,
        )

        # Use different neighborhood sizes for the two levels
        neighborhood_sizes = [3, 5]

        # Call the function
        multilevel_neighborhood_indices, oob_mask, level_indices = (
            get_multilevel_neighborhoods(
                query_positions, level_shapes, neighborhood_sizes
            )
        )

        # Validate output shapes
        n_queries, position_dim = query_positions.shape
        expected_total_elements = 3**3 + 5**3  # 27 + 125 = 152 elements
        assert multilevel_neighborhood_indices.shape == (
            n_queries,
            expected_total_elements,
            position_dim,
        )
        assert level_indices.shape == (expected_total_elements,)

        # Check level indices distribution
        assert torch.all(level_indices[:27] == 0)  # First 3*3*3 elements
        assert torch.all(level_indices[27:] == 1)  # Remaining from level 1

        expected_oob_mask = torch.zeros_like(oob_mask)
        expected_oob_mask[1, level_indices == 1] = (
            multilevel_neighborhood_indices[1, level_indices == 1] >= 6
        ).any(-1)

        assert torch.equal(oob_mask, expected_oob_mask)


@pytest.mark.cuda_if_available
class TestInitialization:
    """Tests for initialization of SparseNeighborhoodAttentionBlock."""

    def test_basic_initialization(self, base_config: dict[str, Any], device: str):
        """Test basic initialization with default parameters."""
        module = SparseNeighborhoodAttentionBlock(**base_config).to(device)

        # Check core attributes
        assert module.embed_dim == base_config["embed_dim"]
        assert module.n_heads == base_config["n_heads"]
        assert module.position_dim == base_config["position_dim"]
        assert module.norm_first == base_config["norm_first"]
        assert module.n_levels == base_config["n_levels"]
        assert torch.equal(
            module.neighborhood_sizes, torch.tensor(base_config["neighborhood_sizes"])
        )

        # Check submodules
        assert isinstance(module.norm, nn.LayerNorm)
        assert isinstance(module.q_in_proj, nn.Linear)
        assert isinstance(module.subset_attn, BatchSparseIndexSubsetAttention)
        assert isinstance(module.pos_encoding, RoPEEncodingND)
        assert isinstance(module.out_proj, nn.Linear)

        # Check dimensions
        assert module.q_in_proj.in_features == base_config["embed_dim"]
        assert module.q_in_proj.out_features == base_config["embed_dim"]
        assert module.out_proj.out_features == base_config["embed_dim"]

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("dropout", 0.2),
            ("bias", True),
            ("norm_first", False),
            ("position_dim", 3),
            ("neighborhood_sizes", [3, 5, 9]),
            ("rope_spatial_base_theta", 1000.0),
            ("rope_level_base_theta", 50.0),
            ("rope_share_heads", True),
            ("rope_freq_group_pattern", "partition"),
        ],
    )
    def test_custom_initialization(
        self,
        base_config: dict[str, Any],
        param_name: str,
        param_value: Any,
        device: str,
    ):
        """Test initialization with custom parameters."""
        config = base_config.copy()
        config[param_name] = param_value
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Check parameter was set correctly
        if param_name == "dropout":
            assert module.attn_drop_rate == param_value
        elif param_name == "bias":
            assert (module.q_in_proj.bias is not None) == param_value
            assert (module.subset_attn.kv_proj.bias is not None) == param_value
        elif param_name == "neighborhood_sizes":
            assert torch.equal(module.neighborhood_sizes, torch.tensor(param_value))
        elif param_name == "rope_freq_group_pattern":
            expected_pattern = get_multilevel_freq_group_pattern(
                config["position_dim"], param_value, device=device
            )
            assert torch.equal(
                module.pos_encoding.freq_group_pattern.bool(), expected_pattern.bool()
            )
        else:
            assert getattr(module, param_name) == param_value

    def test_incompatible_neighborhood_sizes(
        self, base_config: dict[str, Any], device: str
    ):
        """Test initialization with incompatible number of neighborhood sizes."""
        config = base_config.copy()
        config["neighborhood_sizes"] = [3, 5]  # Only 2 sizes, but n_levels=3

        with pytest.raises(ValueError, match=r"Expected len\(neighborhood_sizes\)"):
            _ = SparseNeighborhoodAttentionBlock(**config).to(device)

    def test_even_neighborhood_sizes(self, base_config: dict[str, Any], device: str):
        """Test initialization with even neighborhood sizes."""
        config = base_config.copy()
        config["neighborhood_sizes"] = [2, 4, 6]  # Even sizes

        with pytest.raises(ValueError, match="Expected neighborhood_sizes to be all"):
            SparseNeighborhoodAttentionBlock(**config).to(device)

    @settings(deadline=None)
    @given(inputs=neighborhood_data_strategy())
    def test_initialization_hypothesis(self, inputs: dict[str, Any], device: str):
        """Property-based testing for initialization."""
        config = inputs["config"]

        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        assert module.embed_dim == config["embed_dim"]
        assert module.n_heads == config["n_heads"]
        assert module.position_dim == config["position_dim"]
        assert module.n_levels == config["n_levels"]
        assert module.norm_first == config["norm_first"]

        # Check dimensions
        assert module.q_in_proj.in_features == config["embed_dim"]
        assert module.q_in_proj.out_features == config["embed_dim"]
        assert module.subset_attn.kv_proj.in_features == config["embed_dim"]
        assert module.subset_attn.kv_proj.out_features == 2 * config["embed_dim"]
        assert module.out_proj.out_features == config["embed_dim"]


@pytest.mark.cuda_if_available
class TestForward:
    """Tests for forward pass of SparseNeighborhoodAttentionBlock."""

    def test_forward_shape_preservation(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test that output shape matches input shape."""
        input_data = simple_sparse_input_tensors(device=device)
        query = input_data["query"]

        output = base_module_instance(**input_data)

        # Check output shape and validity
        assert output.shape == query.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_small_inputs(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test forward pass with minimal-sized inputs."""
        # Create minimal input - just one query
        torch.manual_seed(42)
        input_data = simple_sparse_input_tensors(
            device=device,
            n_queries=1,
            embed_dim=base_module_instance.embed_dim,
            position_dim=base_module_instance.position_dim,
        )

        # Run forward pass
        output = base_module_instance(**input_data)

        # Check output
        assert output.shape == input_data["query"].shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("norm_first", [True, False])
    def test_norm_placement(
        self, base_config: dict[str, Any], norm_first: bool, device: str
    ):
        """Test norm_first parameter effect on normalization placement."""
        # Create module with specified norm_first
        config = base_config.copy()
        config["norm_first"] = norm_first
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Create input data
        torch.manual_seed(0)
        input_data = simple_sparse_input_tensors(
            device=device,
            n_queries=4,
            embed_dim=config["embed_dim"],
            position_dim=config["position_dim"],
        )

        # Create hooks to capture intermediate values
        hook = ModuleHook(
            module,
            {
                "q_in_proj": lambda m: m.q_in_proj,
                "norm": lambda m: m.norm,
            },
        )

        # Run forward pass
        with hook, torch.no_grad():
            module(**input_data)

        # Check if input to q_in_proj is normalized based on norm_first
        pre_q_input = hook.captured_values["q_in_proj"]["inputs"]["args"][0]

        if norm_first:
            # Pre-norm: input to q_in_proj should be normalized
            assert torch.abs(pre_q_input.mean()) < 0.1
            assert abs(pre_q_input.std() - 1.0) < 0.5
        else:
            # Post-norm: input to q_in_proj should be the original input
            assert torch.abs(pre_q_input.mean() - input_data["query"].mean()) < 0.1

    @pytest.mark.parametrize("dropout_value", [0.0, 0.3])
    def test_dropout_effect(
        self, base_config: dict[str, Any], dropout_value: float, device: str
    ):
        """Test dropout parameter effect."""
        # Configure module with specified dropout
        config = base_config.copy()
        config["dropout"] = dropout_value
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Generate random input
        torch.manual_seed(0)
        input_data = simple_sparse_input_tensors(
            device=device,
            n_queries=8,
            embed_dim=config["embed_dim"],
            position_dim=config["position_dim"],
        )

        # Output in eval mode (no dropout)
        module.eval()
        eval_output = module(**input_data)

        # Set to training mode
        module.train()

        # Compare outputs with same and different seeds
        torch.manual_seed(1)
        output_1 = module(**input_data)

        torch.manual_seed(1)
        output_2 = module(**input_data)

        torch.manual_seed(2)
        output_3 = module(**input_data)

        # Same seed should give same result
        assert torch.allclose(output_1, output_2)

        # With dropout=0, all outputs should be the same
        # With dropout>0, different seeds should give different results
        if dropout_value == 0.0:
            assert torch.allclose(eval_output, output_1)
            assert torch.allclose(output_1, output_3)
        else:
            assert not torch.allclose(eval_output, output_1)
            assert not torch.allclose(output_1, output_3)

    def test_grad_flow(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test gradient flow through the module."""
        # Make query require grad
        input_tensors = simple_sparse_input_tensors(device=device)
        input_copy = {
            k: (
                v.detach().clone().requires_grad_(True)
                if torch.is_floating_point(v)
                else v.clone()
            )
            for k, v in input_tensors.items()
        }

        # Forward and backward
        output = base_module_instance(**input_copy)
        loss = output.sum()
        loss.backward()

        # Check gradients
        for k, v in input_copy.items():
            if isinstance(v, Tensor) and torch.is_floating_point(v):
                assert v.grad is not None, f"{k} has no grad"
        for name, param in base_module_instance.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    @settings(deadline=None)
    @given(inputs=neighborhood_data_strategy())
    def test_forward_hypothesis(self, inputs: dict[str, Any], device: str):
        """Property-based forward pass test."""
        config = inputs["config"]
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        input_data = strategy_input_tensors(**inputs["tensor_config"], device=device)

        # Run forward pass
        output = module(**input_data)

        # Check output shape and validity
        assert input_data["query"] is not None
        assert output.shape == input_data["query"].shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.cuda_if_available
class TestAttentionMechanism:
    """Tests for attention calculation in SparseNeighborhoodAttentionBlock."""

    def test_position_dependent_output(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test that different positions produce different outputs."""
        # Generate baseline data
        torch.manual_seed(0)
        original_data = simple_sparse_input_tensors(
            device=device,
            n_queries=4,
            position_dim=base_module_instance.position_dim,
            embed_dim=base_module_instance.embed_dim,
        )

        # Create modified data with different positions
        modified_data = original_data.copy()
        modified_positions = original_data["query_spatial_positions"].clone() + 1.0
        modified_data["query_spatial_positions"] = modified_positions

        # Run forward on both
        output_original = base_module_instance(**original_data)
        output_modified = base_module_instance(**modified_data)

        # Outputs should differ due to position-dependent attention
        assert not torch.allclose(output_original, output_modified)

    def test_identical_query_position_attention(
        self, base_config: dict[str, Any], device: str
    ):
        """Test that identical queries at different positions attend differently."""
        module = SparseNeighborhoodAttentionBlock(**base_config).to(device)

        embed_dim = base_config["embed_dim"]

        # Generate data with identical query embeddings but different positions
        input_data = simple_sparse_input_tensors(
            device=device, n_queries=2, random_seed=0
        )

        input_data["query"] = torch.randn(1, embed_dim, device=device).expand(2, -1)
        input_data["query_spatial_positions"] = torch.tensor(
            [
                [5.0, 5.0],
                [10.0, 10.0],
            ],
            device=device,
        )

        # Run forward pass
        output = module(**input_data)

        # Even though queries are identical, outputs should be different due to
        # different neighborhoods being attended to
        assert not torch.allclose(output[0], output[1])


@pytest.mark.cuda_if_available
class TestEdgeCases:
    """Tests for edge cases and validation in SparseNeighborhoodAttentionBlock."""

    def test_dimension_validation(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test validation of input tensor dimensions."""
        # Generate valid data
        input_data = simple_sparse_input_tensors(device=device)

        # Create invalid query (1D instead of 2D)
        invalid_data = input_data.copy()
        invalid_data["query"] = torch.randn(10, device=device)

        # This should raise an error due to incorrect query dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected.*2D"  # type: ignore
        ):
            base_module_instance(**invalid_data)

        # Create invalid position tensor (3D instead of 2D)
        invalid_data = input_data.copy()
        invalid_data["query_spatial_positions"] = torch.randn(2, 3, 2, device=device)

        # This should raise an error due to incorrect position dimensions
        with pytest.raises((ValueError, torch.jit.Error)):  # type: ignore
            base_module_instance(**invalid_data)

    def test_batch_size_mismatch(
        self, base_module_instance: SparseNeighborhoodAttentionBlock, device: str
    ):
        """Test handling of batch size mismatch."""
        # Generate valid data
        input_data = simple_sparse_input_tensors(device=device)

        # Create invalid batch_offsets (wrong number of batches)
        invalid_data = input_data.copy()
        invalid_data["query_batch_offsets"] = torch.tensor([0, 2, 4], device=device)

        # This might raise an error or handle the inconsistency
        try:
            output = base_module_instance(**invalid_data)
            # If it doesn't raise an error, check that output shape is correct
            assert output.shape == input_data["query"].shape
        except (ValueError, torch.jit.Error, IndexError):  # type: ignore
            # It's acceptable for this to raise an error too
            pass

    def test_reset_parameters(self, base_config: dict[str, Any], device: str):
        """Test parameter resetting."""
        module = SparseNeighborhoodAttentionBlock(**base_config).to(device)

        # Store original parameters
        original_q_weight = module.q_in_proj.weight.clone()
        original_kv_weight = module.subset_attn.kv_proj.weight.clone()
        original_out_proj_weight = module.out_proj.weight.clone()

        # Reset parameters
        module.reset_parameters()

        # Parameters should be different
        assert not torch.allclose(original_q_weight, module.q_in_proj.weight)
        assert not torch.allclose(original_kv_weight, module.subset_attn.kv_proj.weight)
        assert not torch.allclose(original_out_proj_weight, module.out_proj.weight)

        # Module should still work
        input_data = simple_sparse_input_tensors(device=device)
        output = module(**input_data)
        assert output.shape == input_data["query"].shape

    def test_empty_batch(self, base_config: dict[str, Any], device: str):
        """Test handling of empty batches."""
        module = SparseNeighborhoodAttentionBlock(**base_config).to(device)

        # Create data with empty first batch
        torch.manual_seed(0)

        input_data = simple_sparse_input_tensors(device=device, n_queries=0)

        # Run forward pass - should handle empty batch gracefully
        output = module(**input_data)
        assert output.shape == input_data["query"].shape


@pytest.mark.cuda_if_available
class TestCorrectness:
    """Tests for correctness of SparseNeighborhoodAttentionBlock."""

    @settings(deadline=None, max_examples=25)
    @given(config=neighborhood_data_strategy())
    def test_residual_connection(self, config: dict[str, Any], device: str):
        """Test that residual connection works correctly."""
        # Create module
        module = SparseNeighborhoodAttentionBlock(**config["config"]).to(device)

        # Create input data
        input_data = strategy_input_tensors(**config["tensor_config"], device=device)
        query = input_data["query"]
        assert query is not None

        # ensure input isn't all zeros
        assume(not torch.allclose(query, torch.zeros_like(query)))

        # Zero out weights to test residual connection
        with torch.no_grad():
            module.q_in_proj.weight.zero_()
            module.subset_attn.kv_proj.weight.zero_()
            module.out_proj.weight.zero_()

            if module.q_in_proj.bias is not None:
                module.q_in_proj.bias.zero_()
            if module.subset_attn.kv_proj.bias is not None:
                module.subset_attn.kv_proj.bias.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.zero_()

        # Make hook to get norm output
        hook = ModuleHook(module, {"norm": lambda m: m.norm})

        # Run forward pass
        with torch.no_grad(), hook:
            output = module(**input_data)

        # With zero weights, output should equal input (for norm_first=True)
        # or be a normalized version of input (for norm_first=False)
        if module.norm_first:
            assert torch.allclose(output, query)
        else:
            # Output should equal norm output
            assert output.shape == query.shape
            assert torch.allclose(output, hook.captured_values["norm"]["outputs"][0])

    def test_neighborhood_attention(self, base_config: dict[str, Any], device: str):
        """Test that neighborhoods are properly attended to."""
        # Create module with smaller neighborhoods for easier testing
        config = base_config.copy()
        config["n_levels"] = 1
        config["neighborhood_sizes"] = [3]  # Just one 3x3 neighborhood
        config["position_dim"] = 2
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Create test data with structured feature map
        torch.manual_seed(0)
        embed_dim = config["embed_dim"]
        h, w = 16, 16

        # Single query with position in the middle of a 16x16 grid
        query = torch.randn(1, embed_dim, device=device)
        query_spatial_positions = torch.tensor([[8.0, 8.0]], device=device)
        query_batch_offsets = torch.tensor([0, 1], device=device)
        level_spatial_shapes = torch.tensor([[h, w]], device=device)

        # Create a feature map where positions closer to the center have higher values
        feature_map = torch.zeros(h, w, embed_dim, device=device)
        center_i, center_j = 8, 8

        # For simplicity, make feature map match query at center, decreasing similarity with distance
        for i in range(h):
            for j in range(w):
                distance = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
                similarity = math.exp(-distance / 5.0)  # Gaussian-like falloff
                feature_map[i, j] = query[0] * similarity

        stacked_feature_maps = feature_map.reshape(1, h, w, 1, embed_dim).to_sparse(
            dense_dim=1
        )

        input_data = {
            "query": query,
            "query_spatial_positions": query_spatial_positions,
            "query_batch_offsets": query_batch_offsets,
            "stacked_feature_maps": stacked_feature_maps,
            "level_spatial_shapes": level_spatial_shapes,
        }

        # Create hook to inspect attention patterns
        hook = ModuleHook(module, {"subset_attn": lambda m: m.subset_attn})

        # Run forward pass
        with torch.no_grad(), hook:
            output = module(**input_data)

        # Get attention index tensor
        key_index_tensor = hook.captured_values["subset_attn"]["inputs"]["args"][1]

        # The output should be more similar to the query than a random vector would be
        query_output_similarity = F.cosine_similarity(query, output)
        assert (
            query_output_similarity > 0.5
        ), f"Expected high similarity, got {query_output_similarity}"

        # Verify neighborhood indices were properly calculated
        # Check that spatial indices in key_index_tensor are within expected range
        # for a 3x3 neighborhood centered at (8, 8)
        spatial_indices_x = key_index_tensor[0, :, 1]
        spatial_indices_y = key_index_tensor[0, :, 2]

        # 3x3 neighborhood around (8, 8) should have x,y in {7,8,9}
        assert torch.all(
            (spatial_indices_x >= 7) & (spatial_indices_x <= 9)
        ), f"Unexpected x indices: {spatial_indices_x}"
        assert torch.all(
            (spatial_indices_y >= 7) & (spatial_indices_y <= 9)
        ), f"Unexpected y indices: {spatial_indices_y}"


@pytest.mark.cuda_if_available
class TestProperties:
    @settings(deadline=None)
    @given(
        inputs=neighborhood_data_strategy(require_grads=True, standard_float_range=True)
    )
    def test_gradient_magnitude_consistency(self, inputs: dict[str, Any], device: str):
        """Test that gradient magnitudes are reasonable and don't explode or vanish."""
        config = inputs["config"]
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Create input data that requires gradients
        input_data = strategy_input_tensors(**inputs["tensor_config"], device=device)

        # Forward pass
        output = module(**input_data)
        loss = output.mean()

        # Backward pass
        loss.backward()

        tensors_requiring_grads = inputs["tensor_config"]["tensors_requiring_grads"]
        for name in tensors_requiring_grads:
            tensor = input_data[name]
            assert tensor is not None
            assert tensor.grad is not None

        # Check gradient magnitudes for inputs
        for name, tensor in input_data.items():
            if (
                not inputs["tensor_config"]["make_level_indices"]
                and name == "query_level_indices"  # Only optional tensor
            ):
                assert tensor is None
            else:
                assert tensor is not None, f"{name} is None"

            if tensor is not None and tensor.grad is not None:
                # Check that gradients are finite
                assert not torch.isnan(tensor.grad).any(), f"{name} has NaN gradients"
                assert not torch.isinf(tensor.grad).any(), f"{name} has Inf gradients"

                # Gradient magnitude should be proportional to tensor magnitude
                if tensor.is_sparse:
                    values = tensor.coalesce().values()
                    grad = tensor.grad.coalesce().values()
                else:
                    values = tensor
                    grad = tensor.grad
                if values.numel() > 0 and values.abs().max() > 1e-6:
                    grad_magnitude_ratio = grad.abs().mean() / values.abs().mean()
                    assert grad_magnitude_ratio < 100.0, (
                        f"{name} gradient too large: "
                        f"magnitude ratio: {grad_magnitude_ratio}"
                    )

        # Check parameter gradients
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"{name} has NaN gradients"
                assert not torch.isinf(param.grad).any(), f"{name} has Inf gradients"

    @settings(deadline=None)
    @given(inputs=neighborhood_data_strategy())
    def test_position_shift_invariance(self, inputs: dict[str, Any], device: str):
        """Test that consistent position shifts produce consistent output changes."""
        # assume nonempty data
        assume(inputs["tensor_config"]["query_batch_offsets"][-1] > 1)
        # assume nonzero data
        assume(
            inputs["tensor_config"]["max_float_value"]
            - inputs["tensor_config"]["min_float_value"]
            > 1e-4
        )
        # assume non-unit key grid
        assume(not (inputs["tensor_config"]["level_spatial_shapes"] == 1).all())
        # assume non-tiny embedding dimension
        assume(inputs["config"]["embed_dim"] > 4)
        # force queries to have something to attend to
        inputs["tensor_config"]["query_full_neighborhood_portion"] = 1.0
        # force dropout inactive
        inputs["config"]["dropout"] = 0.0

        config = inputs["config"]
        module = SparseNeighborhoodAttentionBlock(**config).to(device)

        # Create input data
        input_data = strategy_input_tensors(**inputs["tensor_config"], device=device)

        # Make a copy with shifted positions
        shifted_data = input_data.copy()
        assert shifted_data["query_spatial_positions"] is not None
        assert input_data["level_spatial_shapes"] is not None
        shift_amount = shifted_data["query_spatial_positions"].new_ones(
            config["position_dim"]
        ) * max(1.0, input_data["level_spatial_shapes"].min().item() * 0.05)
        shifted_data["query_spatial_positions"] = (
            input_data["query_spatial_positions"] + shift_amount
        )

        # Run forward on both inputs with hooks to capture attention output
        with torch.no_grad():
            with ModuleHook(
                module, {"subset_attn": lambda m: m.subset_attn}
            ) as hook_original:
                _ = module(**input_data)
            with ModuleHook(
                module, {"subset_attn": lambda m: m.subset_attn}
            ) as hook_shifted:
                _ = module(**shifted_data)

        attn_out_original = hook_original.captured_values["subset_attn"]["outputs"][0]
        attn_out_shifted = hook_shifted.captured_values["subset_attn"]["outputs"][0]

        # Attention outputs should be different but structured
        # We can check:
        # 1. Outputs are different
        assert not torch.allclose(attn_out_original, attn_out_shifted)

        # 2. The relative change is similar across queries in the same batch
        batch_indices = batch_offsets_to_indices(input_data["query_batch_offsets"])
        assert input_data["query_batch_offsets"] is not None
        for batch_idx in range(len(input_data["query_batch_offsets"]) - 1):
            batch_mask = batch_indices == batch_idx
            if batch_mask.sum() >= 2:
                sim_diff = F.cosine_similarity(
                    attn_out_original[batch_mask], attn_out_shifted[batch_mask], dim=1
                )
                # Check similarity variability is relatively small
                if (n := sim_diff.numel()) > 1:
                    threshold = min(0.9, 0.45 * (1.0 + 2.0 / math.sqrt(n)))
                    assert (
                        sim_diff.std() < threshold
                    ), "Position shift should affect outputs consistently"

    @settings(deadline=None)
    @given(inputs=neighborhood_data_strategy())
    def test_neighborhood_size_scaling(self, inputs: dict[str, Any], device: str):
        """Test that neighborhood sizes scale attention contexts correctly."""
        # assume at least 2 queries
        assume(inputs["tensor_config"]["query_batch_offsets"][-1] > 1)
        # assume nonzero data
        assume(
            inputs["tensor_config"]["max_float_value"]
            - inputs["tensor_config"]["min_float_value"]
            > 1e-4
        )
        # assume big enough key grid
        assume((inputs["tensor_config"]["level_spatial_shapes"] >= 5).all())
        # assume non-tiny embedding dimension
        assume(inputs["config"]["embed_dim"] > 4)
        # force non sparse
        inputs["tensor_config"]["query_full_neighborhood_portion"] = 1.0
        inputs["tensor_config"]["sparse_region_sparsity"] = 0.0
        # force dropout inactive
        inputs["config"]["dropout"] = 0.0

        # Create two modules with different neighborhood sizes
        small_nhood_size = 1
        large_nhood_size = 3
        small_nhoods = [small_nhood_size] * inputs["config"]["n_levels"]
        large_nhoods = [large_nhood_size] * inputs["config"]["n_levels"]

        config_small = inputs["config"].copy()
        config_small["neighborhood_sizes"] = small_nhoods

        config_large = inputs["config"].copy()
        config_large["neighborhood_sizes"] = large_nhoods

        # Use same seed to get same parameters
        torch.manual_seed(inputs["tensor_config"]["seed"])
        module_small = SparseNeighborhoodAttentionBlock(**config_small).to(device)

        torch.manual_seed(inputs["tensor_config"]["seed"])
        module_large = SparseNeighborhoodAttentionBlock(**config_large).to(device)

        # Check params are the same
        assert torch.equal(module_small.q_in_proj.weight, module_large.q_in_proj.weight)
        assert torch.equal(module_small.out_proj.weight, module_large.out_proj.weight)
        assert torch.equal(
            module_small.subset_attn.kv_proj.weight,
            module_large.subset_attn.kv_proj.weight,
        )
        for freq_large, freq_small in zip(
            module_large.pos_encoding.freqs, module_small.pos_encoding.freqs
        ):
            assert torch.equal(freq_large, freq_small)
        if inputs["config"]["bias"]:
            assert torch.equal(module_small.q_in_proj.bias, module_large.q_in_proj.bias)
            assert torch.equal(module_small.out_proj.bias, module_large.out_proj.bias)
            assert torch.equal(
                module_small.subset_attn.kv_proj.bias,
                module_large.subset_attn.kv_proj.bias,
            )

        # Create input data
        input_data = strategy_input_tensors(**inputs["tensor_config"], device=device)

        # Create hooks to log attn inputs and outputs
        hook_small = ModuleHook(module_small, {"subset_attn": lambda m: m.subset_attn})
        hook_large = ModuleHook(module_large, {"subset_attn": lambda m: m.subset_attn})

        # Run forward on both
        with torch.no_grad():
            with hook_small:
                _ = module_small(**input_data)
            with hook_large:
                _ = module_large(**input_data)

        # Verify attended keys are different
        values_small = hook_small.captured_values
        values_large = hook_large.captured_values
        small_key_index_tensor = values_small["subset_attn"]["inputs"]["args"][1]
        large_key_index_tensor = values_large["subset_attn"]["inputs"]["args"][1]

        # Check that neighborhood sizes are correct
        assert small_key_index_tensor.shape[1] < large_key_index_tensor.shape[1]
        assert small_key_index_tensor.shape[1] == sum(
            np.pow(small_nhoods, inputs["config"]["position_dim"])
        )
        assert large_key_index_tensor.shape[1] == sum(
            np.pow(large_nhoods, inputs["config"]["position_dim"])
        )

        small_is_specified_mask = values_small["subset_attn"]["outputs"][1]
        large_is_specified_mask = values_large["subset_attn"]["outputs"][1]

        small_specified_indices = small_key_index_tensor[small_is_specified_mask]
        large_specified_indices = large_key_index_tensor[large_is_specified_mask]
        assert (
            small_specified_indices.shape[0] < large_specified_indices.shape[0]
        ), f"query positions: {input_data['query_spatial_positions']}"

        # Test that nonzero attended keys are different
        small_specified_unique = small_specified_indices.unique(dim=0)
        large_specified_unique = large_specified_indices.unique(dim=0)
        assert not torch.equal(small_specified_unique, large_specified_unique)
        assert small_specified_unique.shape[0] < large_specified_unique.shape[0]

        # Test that actual attention outputs are different
        attn_out_small = values_small["subset_attn"]["outputs"][0]
        attn_out_large = values_large["subset_attn"]["outputs"][0]
        assert not torch.allclose(attn_out_small, attn_out_large), (
            f"Output max diff: {(attn_out_small - attn_out_large).abs().max()}"
            " total additional keys: "
            f"{large_specified_unique.shape[0] - small_specified_unique.shape[0]}"
        )

    @settings(deadline=None)
    @given(inputs=neighborhood_data_strategy(standard_float_range=True))
    def test_multi_level_contribution(self, inputs: dict[str, Any], device: str):
        """Test that multiple levels contribute to the output."""
        # assume at least 2 queries
        assume(inputs["tensor_config"]["query_batch_offsets"][-1] > 1)
        # assume nonzero data
        assume(
            inputs["tensor_config"]["max_float_value"]
            - inputs["tensor_config"]["min_float_value"]
            > 1e-4
        )
        # assume more than 1 feature level
        assume(inputs["config"]["n_levels"] > 1)
        # assume non-tiny embedding dimension
        assume(inputs["config"]["embed_dim"] > 4)
        # force queries to have something to attend to
        inputs["tensor_config"]["query_full_neighborhood_portion"] = 1.0
        inputs["tensor_config"]["sparse_region_sparsity"] = 0.0
        # force dropout inactive
        inputs["config"]["dropout"] = 0.0

        assume(inputs["config"]["n_levels"] > 1)  # Skip test if only one level

        module = SparseNeighborhoodAttentionBlock(**inputs["config"]).to(device)

        # Create input data
        input_data = strategy_input_tensors(**inputs["tensor_config"], device=device)
        sparse_tensor = input_data["stacked_feature_maps"]
        assert sparse_tensor is not None
        original_indices = sparse_tensor.indices()

        # Run with all levels
        with torch.no_grad():
            all_output = module(**input_data)

        # Check if at least one single-level output differs from all-level output
        level_outputs = []
        for level in range(inputs["config"]["n_levels"]):
            # Create sparse tensor with just this level
            level_mask = original_indices[-1] == level

            level_data = input_data.copy()
            level_data["stacked_feature_maps"] = torch.sparse_coo_tensor(
                original_indices[:, level_mask],
                sparse_tensor.values()[level_mask],
                sparse_tensor.size(),
                device=device,
            ).coalesce()

            # Run forward with only this level
            with torch.no_grad():
                level_outputs.append(module(**level_data))

        # At least one level should produce different output than all levels combined
        assert any(
            [not torch.allclose(output, all_output) for output in level_outputs]
        ), (
            "Multi-level output should differ from single-level outputs. Output max diffs:"
            f"{[(output - all_output).abs().max() for output in level_outputs]}, "
            "not allclose: "
            f"{[not torch.allclose(output, all_output) for output in level_outputs]}"
            ", cosine sims: "
            f"{[F.cosine_similarity(output, all_output) for output in level_outputs]}"
        )

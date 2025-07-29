import math
from typing import Any, Union, Optional

import numpy as np
import pytest
import torch
import hypothesis
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from torch import Tensor, nn

from nd_rotary_encodings import (
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
)
from sparse_transformer_layers.blocks.self_attn import (
    MultilevelSelfAttentionBlockWithRoPE,
)
from .conftest import ModuleHook
from ..conftest import positions_strategy


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for MultilevelSelfAttentionBlockWithRoPE tests."""
    return {
        "embed_dim": 64,
        "n_heads": 4,
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
def module_instance(
    base_config: dict[str, Any], device: str
) -> MultilevelSelfAttentionBlockWithRoPE:
    """Create a module instance for testing."""
    return MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)


@st.composite
def input_data_strategy(draw) -> dict[str, Any]:
    # Draw basic shape parameters
    n_heads = draw(st.integers(1, 8))
    head_dim = draw(st.integers(1, 16)) * 2
    embed_dim = n_heads * head_dim
    position_dim = draw(st.integers(1, 5))
    n_levels = draw(st.integers(1, 5))

    # Draw data size parameters
    seq_lengths = draw(st.lists(st.integers(0, 32), min_size=1, max_size=4))
    max_seq_len = max(seq_lengths)
    batch_size = len(seq_lengths)
    batch_offsets = np.zeros((batch_size + 1,), dtype=int)
    batch_offsets[1:] = np.cumsum(seq_lengths)
    total_seq_lengths = batch_offsets[-1].item()

    # Draw level info
    level_indices = draw(
        st.lists(
            elements=st.integers(0, n_levels - 1),
            min_size=total_seq_lengths,
            max_size=total_seq_lengths,
        )
    )
    level_shapes = draw(
        arrays(
            np.int64,
            (n_levels, position_dim),
            elements=st.integers(1, 1000),
            fill=st.nothing(),
        )
    )

    # Draw tensor generation parameters
    seed = draw(st.integers(min_value=0, max_value=int(1e8)))
    float_dtype = draw(st.just(torch.float32))

    min_float_value = draw(st.floats(min_value=-1e6, max_value=1e6, exclude_max=True))
    max_float_value = draw(st.floats(min_value=min_float_value, max_value=1e6))

    # Generation params for attn_mask
    attn_mask_sparsity = draw(st.floats(0.0, 1.0))
    attn_mask_shape = draw(
        st.sampled_from(
            [
                (batch_size, max_seq_len, max_seq_len),
                (batch_size * n_heads, max_seq_len, max_seq_len),
                (batch_size, n_heads, max_seq_len, max_seq_len),
            ]
        )
    )

    position_dtype, min_position, max_position = draw(
        positions_strategy().filter(lambda drawn: drawn[1] <= 0 or drawn[2] >= 1)
    )

    # Select rope freq group pattern and determine if nonequal freq groups need to
    # be allowed
    rope_freq_group_pattern = draw(st.sampled_from(["single", "partition", "closure"]))
    if rope_freq_group_pattern == "partition":
        rope_enforce_freq_groups_equal = head_dim % (2 * 2) == 0
    elif rope_freq_group_pattern == "closure":
        rope_enforce_freq_groups_equal = head_dim % (2 * 3) == 0
    else:
        rope_enforce_freq_groups_equal = True

    return {
        "config": {
            "embed_dim": embed_dim,
            "n_heads": n_heads,
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
            "batch_offsets": batch_offsets,
            "level_indices": level_indices,
            "level_spatial_shapes": level_shapes,
            "min_float_value": min_float_value,
            "max_float_value": max_float_value,
            "min_position": min_position,
            "max_position": max_position,
            "float_dtype": float_dtype,
            "position_dtype": position_dtype,
            "attn_mask_sparsity": attn_mask_sparsity,
            "attn_mask_shape": attn_mask_shape,
            "seed": seed,
        },
        "extra": {
            "seq_lengths": seq_lengths,
        },
    }


def make_input_tensors(
    embed_dim: int,
    position_dim: int,
    batch_offsets: list[int],
    level_indices: list[int],
    level_spatial_shapes: np.ndarray,
    min_float_value: float,
    max_float_value: float,
    min_position: Union[int, float],
    max_position: Union[int, float],
    float_dtype: torch.dtype,
    position_dtype: torch.dtype,
    attn_mask_sparsity: float,
    attn_mask_shape: tuple[int, ...],
    seed: int,
    device: Union[str, torch.device],
    force_non_normalized_coords: bool = True,
) -> dict[str, Tensor]:
    if isinstance(device, str):
        device = torch.device(device, 0)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    total_seq_lengths = batch_offsets[-1]

    # embeddings
    x = torch.empty((total_seq_lengths, embed_dim), device=device, dtype=float_dtype)
    x.uniform_(min_float_value, max_float_value)

    # spatial positions
    spatial_positions: torch.Tensor = torch.empty(
        (total_seq_lengths, position_dim), device=device, dtype=position_dtype
    )
    if position_dtype == torch.long:
        assert isinstance(min_position, int)
        spatial_positions.random_(min_position, min_position + 1)
    else:
        spatial_positions.uniform_(min_position, max_position)
    if force_non_normalized_coords and spatial_positions.numel() > 0:
        while spatial_positions.min() > 0.0 and spatial_positions.max() <= 1.0:
            spatial_positions = spatial_positions + 1.0

    # attn_mask: True means mask out
    attn_mask = torch.empty(attn_mask_shape, dtype=torch.bool, device=device)
    attn_mask.bernoulli_(attn_mask_sparsity)

    # reset rng state
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state, device)
    else:
        torch.set_rng_state(rng_state)

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": torch.tensor(level_indices, device=device),
        "level_spatial_shapes": torch.tensor(level_spatial_shapes, device=device),
        "batch_offsets": torch.tensor(batch_offsets, device=device),
        "attn_mask": attn_mask,
    }


@pytest.fixture(params=["simple", "complex", "variable_length"])
def input_data(request, device: str) -> dict[str, Optional[torch.Tensor]]:
    """Parametrized fixture generating different input data types."""
    if request.param == "simple":
        return generate_input_data(
            device=device,
            stacked_sequence_length=6,
            embed_dim=64,
            position_dim=2,
            batch_size=2,
            num_levels=2,
            level_distribution=[0, 0, 1, 1, 1, 0],
            level_shapes=[[8, 8], [4, 4]],
            random_seed=42,
        )
    elif request.param == "complex":
        return generate_input_data(
            device=device,
            stacked_sequence_length=12,
            embed_dim=64,
            position_dim=2,
            batch_size=3,
            num_levels=3,
            level_shapes=[[16, 16], [8, 8], [4, 4]],
            with_attn_mask=True,
            random_seed=43,
        )
    elif request.param == "variable_length":
        return generate_variable_length_input_data(
            device=device,
            embed_dim=64,
            position_dim=2,
            num_levels=3,
            level_shapes=[[16, 16], [8, 8], [4, 4]],
            tokens_per_level=[
                [16, 8, 4],  # Batch 1
                [12, 6, 3],  # Batch 2 (smaller)
                [20, 10, 5],  # Batch 3 (larger)
            ],
            random_seed=44,
        )
    else:
        raise ValueError(f"Unrecognized param {request.param}")


def generate_input_data(
    device: str,
    stacked_sequence_length: int,
    embed_dim: int,
    position_dim: int,
    batch_size: int,
    num_levels: int,
    level_distribution: Optional[list[int]] = None,
    level_shapes: Optional[list[list[int]]] = None,
    tokens_per_batch: Optional[list[int]] = None,
    with_attn_mask: bool = False,
    random_seed: Optional[int] = None,
) -> dict[str, Optional[torch.Tensor]]:
    """Generate input tensors for testing with customizable parameters."""
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Create input embedding tensor
    x = torch.randn(stacked_sequence_length, embed_dim, device=device)

    # Create spatial positions
    spatial_positions = (
        torch.rand(stacked_sequence_length, position_dim, device=device) * 10
    )

    # Create level indices
    if level_distribution is not None:
        level_indices = torch.tensor(level_distribution, device=device)
    else:
        # Generate distribution with roughly equal representation of each level
        level_indices = torch.randint(
            0, num_levels, (stacked_sequence_length,), device=device
        )

    # Create level spatial shapes
    if level_shapes is not None:
        level_spatial_shapes = torch.tensor(level_shapes, device=device)
    else:
        # Generate decreasing shapes for each level
        base_size = 16
        level_spatial_shapes = torch.tensor(
            [[base_size // (2**i), base_size // (2**i)] for i in range(num_levels)],
            device=device,
        )

    # Create batch offsets
    if tokens_per_batch is not None:
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        batch_offsets[1:] = torch.tensor(tokens_per_batch, device=device).cumsum(0)
    else:
        # Equal distribution of tokens across batches
        tokens = stacked_sequence_length // batch_size
        batch_offsets = torch.arange(
            0, stacked_sequence_length + 1, tokens, device=device
        )
        if batch_offsets.size(0) > batch_size + 1:
            batch_offsets = batch_offsets[: batch_size + 1]
        batch_offsets[-1] = stacked_sequence_length

    # Create attention mask if requested
    attn_mask = None
    if with_attn_mask:
        # Create simple mask: first token in each batch can't attend to last token
        tokens_per_batch = [
            int(batch_offsets[i + 1] - batch_offsets[i]) for i in range(batch_size)
        ]
        max_tokens = max(tokens_per_batch)
        attn_mask = torch.zeros(
            batch_size, max_tokens, max_tokens, dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            if tokens_per_batch[i] > 1:
                attn_mask[i, 0, tokens_per_batch[i] - 1] = True

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "batch_offsets": batch_offsets,
        "attn_mask": attn_mask,
    }


def generate_variable_length_input_data(
    device: str,
    embed_dim: int,
    position_dim: int,
    num_levels: int,
    level_shapes: list[list[int]],
    tokens_per_level: list[list[int]],
    random_seed: Optional[int] = None,
) -> dict[str, Optional[torch.Tensor]]:
    """Generate input data with variable sequence lengths."""
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Basic setup
    batch_size = len(tokens_per_level)
    total_seq_length = sum(sum(batch) for batch in tokens_per_level)
    level_spatial_shapes = torch.tensor(level_shapes, device=device)

    # Calculate batch offsets
    batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    batch_offsets[1:] = torch.cumsum(
        torch.tensor([sum(batch) for batch in tokens_per_level], device=device), 0
    )

    # Create embeddings
    x = torch.randn(total_seq_length, embed_dim, device=device)

    # Create spatial positions and level indices
    spatial_positions = torch.zeros(total_seq_length, position_dim, device=device)
    level_indices = torch.zeros(total_seq_length, dtype=torch.long, device=device)

    # Fill spatial positions and level indices
    idx = 0
    for b in range(batch_size):
        for level in range(num_levels):
            num_tokens = tokens_per_level[b][level]
            h, w = level_spatial_shapes[level]

            # Generate positions on a grid
            if num_tokens > 0:
                h_use = torch.minimum(
                    h, num_tokens // w + (1 if num_tokens % w > 0 else 0)
                )
                positions = []
                for i in range(h_use):
                    for j in range(torch.minimum(w, num_tokens - i * w)):
                        positions.append([i, j])

                # For random positions instead of grid:
                # positions = torch.rand(num_tokens, position_dim, device=device) * torch.tensor([h, w], device=device)

                positions = torch.tensor(positions, device=device)
                num_positions = positions.shape[0]

                # Add to spatial positions and set level indices
                spatial_positions[idx : idx + num_positions] = positions
                level_indices[idx : idx + num_positions] = level

                idx += num_positions

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "batch_offsets": batch_offsets,
        "attn_mask": None,
    }


@pytest.mark.cuda_if_available
class TestInitialization:
    """Tests for initialization of MultilevelSelfAttentionBlockWithRoPE."""

    def test_basic_initialization(self, base_config: dict[str, Any], device: str):
        """Test basic initialization with default parameters."""
        module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)

        # Check core attributes
        assert module.embed_dim == base_config["embed_dim"]
        assert module.n_heads == base_config["n_heads"]
        assert module.position_dim == base_config["position_dim"]
        assert module.norm_first == base_config["norm_first"]

        # Check submodules
        assert isinstance(module.norm, nn.LayerNorm)
        assert isinstance(module.qkv, nn.Linear)
        assert isinstance(module.pos_encoding, RoPEEncodingND)
        assert isinstance(module.out_proj, nn.Linear)

        # Check dimensions
        assert module.qkv.in_features == base_config["embed_dim"]
        assert module.qkv.out_features == 3 * base_config["embed_dim"]
        assert module.out_proj.out_features == base_config["embed_dim"]

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("dropout", 0.2),
            ("bias", True),
            ("norm_first", False),
            ("position_dim", 3),
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
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Check parameter was set correctly
        if param_name == "dropout":
            assert module.attn_drop_rate == param_value
        elif param_name == "bias":
            assert (module.qkv.bias is not None) == param_value
        elif param_name == "rope_freq_group_pattern":
            expected_pattern = get_multilevel_freq_group_pattern(
                config["position_dim"], param_value
            )
            pos_encoding_pattern = module.pos_encoding.freq_group_pattern
            assert pos_encoding_pattern.shape == expected_pattern.shape
        else:
            assert getattr(module, param_name) == param_value

    def test_embed_dim_head_compatibility(self, device: str):
        """Test that initialization handles embed_dim not divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible by"):
            MultilevelSelfAttentionBlockWithRoPE(
                embed_dim=65,  # Not divisible by 4
                n_heads=4,
            ).to(device)

    @settings(deadline=None)
    @given(inputs=input_data_strategy())
    def test_initialization_hypothesis(self, inputs: dict[str, Any], device):
        config = inputs["config"]

        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        assert module.embed_dim == config["embed_dim"]
        assert module.n_heads == config["n_heads"]
        assert module.position_dim == config["position_dim"]
        assert module.norm_first == config["norm_first"]

        # Check submodules
        assert isinstance(module.norm, nn.LayerNorm)
        assert isinstance(module.qkv, nn.Linear)
        assert isinstance(module.pos_encoding, RoPEEncodingND)
        assert isinstance(module.out_proj, nn.Linear)

        # Check dimensions
        assert module.qkv.in_features == config["embed_dim"]
        assert module.qkv.out_features == 3 * config["embed_dim"]
        assert module.out_proj.out_features == config["embed_dim"]


@pytest.mark.cuda_if_available
class TestForward:
    """Tests for forward pass of MultilevelSelfAttentionBlockWithRoPE."""

    def test_forward_shape_preservation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        input_data: dict[str, torch.Tensor],
    ):
        """Test that output shape matches input shape."""
        x = input_data["x"]
        output = module_instance(**input_data)

        # Check output shape and validity
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_small_inputs(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test forward pass with minimal-sized inputs."""
        # Create minimal random inputs - just one token per batch
        torch.manual_seed(42)
        batch_size = 2
        input_data = generate_input_data(
            device=device,
            stacked_sequence_length=batch_size,
            embed_dim=module_instance.embed_dim,
            position_dim=module_instance.position_dim,
            batch_size=batch_size,
            num_levels=1,
            level_shapes=[[1, 1]],
            random_seed=42,
        )

        # Run forward pass
        output = module_instance(**input_data)

        # Check output
        assert input_data["x"] is not None
        assert output.shape == input_data["x"].shape
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
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Create input data
        torch.manual_seed(0)
        input_data = generate_input_data(
            device=device,
            stacked_sequence_length=6,
            embed_dim=config["embed_dim"],
            position_dim=config["position_dim"],
            batch_size=2,
            num_levels=1,
            random_seed=0,
        )

        # Create hook to capture intermediate valuesw
        hook = ModuleHook(
            module,
            {
                "qkv": lambda m: m.qkv,
            },
        )

        # Run forward pass
        with hook, torch.no_grad():
            module(**input_data)

        # Check if input to qkv is normalized based on norm_first
        pre_qkv_input = hook.captured_values["qkv"]["inputs"]["args"][0]

        if norm_first:
            # Pre-norm: input to QKV should be normalized
            assert torch.abs(pre_qkv_input.mean()) < 0.1
            assert abs(pre_qkv_input.std() - 1.0) < 0.5
        else:
            # Post-norm: input to QKV should be the original input
            assert input_data["x"] is not None
            assert torch.abs(pre_qkv_input.mean() - input_data["x"].mean()) < 0.1

    @pytest.mark.parametrize("dropout_value", [0.0, 0.3])
    def test_dropout_effect(
        self, base_config: dict[str, Any], dropout_value: float, device: str
    ):
        """Test dropout parameter effect."""
        # Configure module with specified dropout
        config = base_config.copy()
        config["dropout"] = dropout_value
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Generate random input
        torch.manual_seed(0)
        input_data = generate_input_data(
            device=device,
            stacked_sequence_length=8,
            embed_dim=config["embed_dim"],
            position_dim=config["position_dim"],
            batch_size=2,
            num_levels=1,
            random_seed=0,
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
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        input_data: dict[str, torch.Tensor],
    ):
        """Test gradient flow through the module."""
        # Make inputs require grad
        x = input_data["x"].clone().requires_grad_(True)
        input_copy = input_data.copy()
        input_copy["x"] = x

        # Forward and backward
        output = module_instance(**input_copy)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        for name, param in module_instance.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.filterwarnings("ignore:^Expected un-normalized:UserWarning")
    @settings(deadline=None)
    @given(inputs=input_data_strategy())
    def test_forward_hypothesis(self, inputs, device: str):
        module = MultilevelSelfAttentionBlockWithRoPE(**inputs["config"]).to(device)
        inputs = make_input_tensors(**inputs["tensor_config"], device=device)

        out = module(**inputs)

        assert out.shape == inputs["x"].shape
        assert not out.isnan().any()
        assert not out.isinf().any()


@pytest.mark.cuda_if_available
class TestAttentionMechanism:
    """Tests for attention calculation and masking."""

    def test_attention_mask_effect(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test that attention mask correctly affects output."""
        # Generate data with attention mask
        torch.manual_seed(0)
        data_with_mask = generate_input_data(
            device=device,
            stacked_sequence_length=12,
            embed_dim=module_instance.embed_dim,
            position_dim=module_instance.position_dim,
            batch_size=3,
            num_levels=2,
            with_attn_mask=True,
            random_seed=0,
        )

        # Run with mask
        output_with_mask = module_instance(**data_with_mask)

        # Run without mask
        data_without_mask = data_with_mask.copy()
        data_without_mask["attn_mask"] = None
        output_without_mask = module_instance(**data_without_mask)

        # Outputs should differ with mask
        assert not torch.allclose(output_with_mask, output_without_mask)

    def test_different_position_encoding(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test that different positions produce different outputs."""
        # Generate baseline data
        torch.manual_seed(0)
        original_data = generate_input_data(
            device=device,
            stacked_sequence_length=6,
            embed_dim=module_instance.embed_dim,
            position_dim=module_instance.position_dim,
            batch_size=2,
            num_levels=2,
            random_seed=0,
        )

        # Create modified data with different positions
        modified_data = original_data.copy()
        assert original_data["spatial_positions"] is not None
        modified_positions = original_data["spatial_positions"].clone() + 1.0
        modified_data["spatial_positions"] = modified_positions

        # Run forward on both
        output_original = module_instance(**original_data)
        output_modified = module_instance(**modified_data)

        # Outputs should differ due to position encoding
        assert not torch.allclose(output_original, output_modified)

    @pytest.mark.parametrize("rope_freq_pattern", ["single", "partition", "closure"])
    def test_rope_frequency_patterns(
        self, base_config: dict[str, Any], rope_freq_pattern: str, device: str
    ):
        """Test different RoPE frequency patterns."""
        # Create module with specified pattern
        config = base_config.copy()
        config["rope_freq_group_pattern"] = rope_freq_pattern
        if rope_freq_pattern == "closure":
            config["rope_enforce_freq_groups_equal"] = False
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Generate input data
        torch.manual_seed(0)
        input_data = generate_input_data(
            device=device,
            stacked_sequence_length=8,
            embed_dim=config["embed_dim"],
            position_dim=config["position_dim"],
            batch_size=2,
            num_levels=2,
            random_seed=0,
        )

        # Run forward pass
        output = module(**input_data)

        # Compare with default pattern if not using default
        if rope_freq_pattern != "single":
            default_module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(
                device
            )
            default_output = default_module(**input_data)
            assert not torch.allclose(output, default_output)


@pytest.mark.cuda_if_available
class TestEdgeCases:
    """Tests for edge cases and validation."""

    def test_dimension_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        input_data: dict[str, torch.Tensor],
    ):
        """Test validation of input tensor dimensions."""
        # Create invalid x (1D instead of 2D)
        invalid_data = input_data.copy()
        invalid_data["x"] = torch.randn(10, device=input_data["x"].device)

        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected.*2D"  # type: ignore
        ):
            module_instance(**invalid_data)

    def test_empty_batch_handling(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test handling of empty batches."""
        # Generate data with an empty batch in the middle
        torch.manual_seed(0)
        x = torch.randn(6, module_instance.embed_dim, device=device)
        spatial_positions = torch.rand(6, module_instance.position_dim, device=device)
        level_indices = torch.zeros(6, dtype=torch.long, device=device)
        level_spatial_shapes = torch.tensor([[8, 8]], device=device)

        # Batch offsets: [0, 3, 3, 6] - middle batch is empty
        batch_offsets = torch.tensor([0, 3, 3, 6], device=device)

        # Run forward pass
        output = module_instance(
            x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
        )

        # Check output
        assert output.shape == x.shape

    def test_reset_parameters(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        input_data: dict[str, torch.Tensor],
    ):
        """Test parameter resetting."""
        # Store original parameters
        original_qkv_weight = module_instance.qkv.weight.clone()
        original_out_proj_weight = module_instance.out_proj.weight.clone()

        # Reset parameters
        module_instance.reset_parameters()

        # Parameters should be different
        assert not torch.allclose(original_qkv_weight, module_instance.qkv.weight)
        assert not torch.allclose(
            original_out_proj_weight, module_instance.out_proj.weight
        )

        # Module should still work
        output = module_instance(**input_data)
        assert output.shape == input_data["x"].shape


@pytest.mark.cuda_if_available
class TestCorrectness:
    """Tests that verify correctness of MultilevelSelfAttentionBlockWithRoPE."""

    def test_rope_encoding_scaling(self, base_config: dict, device: str):
        """Verify RoPE encoding applies expected rotations based on positions."""
        # Modify config for a simpler test
        config = base_config.copy()
        config.update(
            {
                "position_dim": 1,  # Simplify to 1D positions
                "rope_spatial_base_theta": 10.0,
                "rope_share_heads": True,
            }
        )

        # Create module
        torch.manual_seed(0)
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Create hook to capture QKV outputs and rotated Q/K
        hook = ModuleHook(
            module,
            {
                "pos_encoding": lambda m: m.pos_encoding,
                "qkv": lambda m: m.qkv,
            },
        )

        # Generate input with same embedding at each location
        seq_len = 8
        embed_dim = config["embed_dim"]
        x = torch.randn(1, embed_dim, device=device).expand(seq_len, -1)

        # Create linearly spaced positions
        spatial_positions = (
            torch.arange(0, seq_len, device=device).view(seq_len, 1).float()
        )
        level_indices = torch.zeros(seq_len, dtype=torch.long, device=device)
        level_spatial_shapes = torch.tensor([[seq_len]], device=device)
        batch_offsets = torch.tensor([0, seq_len], device=device)

        # Run forward pass
        with hook:
            _ = module(
                x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
            )

        # Get the original QKV and rotated Q/K
        qkv_output = hook.captured_values["qkv"]["outputs"][0]
        q, _, _ = qkv_output.chunk(3, dim=-1)

        # Extract pos_encoding inputs and outputs
        prepped_positions = hook.captured_values["pos_encoding"]["inputs"]["args"][1]

        # Check that positions had levels appended
        assert torch.allclose(
            prepped_positions[:, 0], spatial_positions.squeeze(-1).float()
        )
        assert torch.allclose(prepped_positions[:, 1], level_indices.float())

        pos_encoding_outputs = hook.captured_values["pos_encoding"]["outputs"]
        q_rotated = pos_encoding_outputs[0]

        # Position 0 should not be rotated
        pos_0 = (spatial_positions == 0.0).squeeze(1)
        assert torch.equal(q[pos_0], q_rotated[pos_0])

        # Other positions should be rotated
        non_pos_0 = (~pos_0).nonzero()
        for pos in non_pos_0:
            assert not torch.allclose(q[pos], q_rotated[pos])

        # Rotation should not change the norm of the embeddings
        assert torch.allclose(
            torch.linalg.vector_norm(q, dim=1),
            torch.linalg.vector_norm(q_rotated, dim=1),
        )

    def test_attention_pattern(self, base_config: dict, device: str):
        """Test that specific input patterns produce expected attention patterns."""
        # Create a module to test
        module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)

        # Modify the module to capture attention weights
        # We'll create a new method that stores the attention matrix
        attention_weights = []
        original_calc_attn = module._calc_attn

        def _calc_attn_with_capture(*args, **kwargs):
            # Calculate attn normally
            output = original_calc_attn(*args, **kwargs)

            # Extract attention pattern (q @ k.T * scale before softmax)
            q, k = args[0], args[1]
            scale = 1.0 / math.sqrt(q.size(-1))
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Store for inspection
            attention_weights.append(attn.detach().clone())

            return output

        module._calc_attn = _calc_attn_with_capture

        # Create input where token 0 should attend strongly to token 3
        # We'll make token 3 have a pattern very similar to token 0
        torch.manual_seed(40)
        seq_len = 4
        position_dim = base_config["position_dim"]
        embed_dim = base_config["embed_dim"]

        x = torch.randn(seq_len, embed_dim, device=device)
        # Make token 3 similar to token 0 for strong attention
        x[3] = x[0] * 0.9 + torch.randn_like(x[0]) * 0.1

        spatial_positions = torch.zeros(seq_len, position_dim, device=device)
        level_indices = torch.zeros(seq_len, dtype=torch.long, device=device)
        level_spatial_shapes = torch.ones(
            1, position_dim, dtype=torch.long, device=device
        )
        batch_offsets = torch.tensor([0, seq_len], device=device)

        # Run forward pass
        module(x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets)

        # Check that token 0 attends strongly to token 3
        attn_matrix = attention_weights[0]  # (batch, heads, seq, seq)
        attn_weights = torch.softmax(attn_matrix, dim=-1)
        attn_weights_token0 = attn_weights[
            0, :, 0, :
        ]  # attention from token 0 across heads

        # Get average attention weight from token 0 to token 3 across heads
        avg_attn_0_to_3 = attn_weights_token0[:, 3].mean().item()

        # This should be relatively high due to our construction
        assert (
            avg_attn_0_to_3 > 0.25
        ), f"Expected strong attention from token 0 to token 3, got {avg_attn_0_to_3}"

        # Restore original method
        module._calc_attn = original_calc_attn

    @settings(deadline=None)
    @given(inputs=input_data_strategy())
    def test_norm_first_correctness(self, inputs: dict[str, Any], device: str):
        """Verify norm_first parameter works correctly."""

        # ensure nonzero input tensor
        hypothesis.assume(
            abs(inputs["tensor_config"]["min_float_value"]) > 1e-6
            or abs(inputs["tensor_config"]["max_float_value"]) > 1e-6
        )
        # guard against dropout dropping out everything
        hypothesis.assume(inputs["config"]["dropout"] < 0.6)

        # Create input tensors
        input_tensors = make_input_tensors(**inputs["tensor_config"], device=device)

        # ensure that attention mask doesn't mask out everything
        hypothesis.assume(not input_tensors["attn_mask"].all())

        # Create two identical modules except for norm_first setting
        config_norm_first = inputs["config"].copy()
        config_norm_first["norm_first"] = True

        config_norm_last = inputs["config"].copy()
        config_norm_last["norm_first"] = False

        norm_first_module = MultilevelSelfAttentionBlockWithRoPE(
            **config_norm_first
        ).to(device)
        norm_last_module = MultilevelSelfAttentionBlockWithRoPE(**config_norm_last).to(
            device
        )

        with torch.no_grad():
            # Copy weights from first module to second
            norm_last_module.qkv.weight.copy_(norm_first_module.qkv.weight)
            norm_last_module.out_proj.weight.copy_(norm_first_module.out_proj.weight)
            norm_last_module.norm.weight.copy_(norm_first_module.norm.weight)
            norm_last_module.norm.bias.copy_(norm_first_module.norm.bias)

            if (
                hasattr(norm_first_module.qkv, "bias")
                and norm_first_module.qkv.bias is not None
            ):
                norm_last_module.qkv.bias.copy_(norm_first_module.qkv.bias)
            if (
                hasattr(norm_first_module.out_proj, "bias")
                and norm_first_module.out_proj.bias is not None
            ):
                norm_last_module.out_proj.bias.copy_(norm_first_module.out_proj.bias)

        # Create hooks to capture intermediate values
        hook_first = ModuleHook(
            norm_first_module,
            {
                "out_proj": lambda m: m.out_proj,
                "norm": lambda m: m.norm,
                "qkv": lambda m: m.qkv,
            },
        )

        hook_last = ModuleHook(
            norm_last_module,
            {
                "out_proj": lambda m: m.out_proj,
                "norm": lambda m: m.norm,
                "qkv": lambda m: m.qkv,
            },
        )

        # Run forward pass on both modules
        with torch.no_grad():
            with hook_first:
                output_first = norm_first_module(**input_tensors)
            with hook_last:
                output_last = norm_last_module(**input_tensors)

        # Only examine tensors if they are nonempty
        n = input_tensors["x"].numel()
        if n > 0:

            # Check that outputs are different
            assert not torch.allclose(
                output_first, output_last
            ), "Expected different outputs for norm_first vs norm_last"

            # For norm_first, norm should be applied before QKV
            norm_first_norm_out = hook_first.captured_values["norm"]["outputs"][0]
            norm_first_qkv_in = hook_first.captured_values["qkv"]["inputs"]["args"][0]
            norm_first_out_proj = hook_first.captured_values["out_proj"]["outputs"][0]

            # For norm_last, norm should be applied after attention
            norm_last_norm_out = hook_last.captured_values["norm"]["outputs"][0]
            norm_last_qkv_in = hook_last.captured_values["qkv"]["inputs"]["args"][0]
            norm_last_out_proj = hook_last.captured_values["out_proj"]["outputs"][0]

            # Verify norm layers have different outputs as long as non-residual branch
            # is nonzero
            if not torch.allclose(
                norm_first_out_proj, torch.zeros_like(norm_first_norm_out)
            ) and not torch.allclose(
                norm_last_out_proj, torch.zeros_like(norm_last_out_proj)
            ):
                assert not torch.allclose(norm_first_norm_out, norm_last_norm_out)

            # For norm_first, check norm is directly before qkv
            assert torch.allclose(norm_first_norm_out, norm_first_qkv_in)

            # For norm_last, the QKV input should match the original x
            assert torch.allclose(
                norm_last_qkv_in, input_tensors["x"]
            ), "Expected QKV input to match original x for norm_last"

    @settings(deadline=None)
    @given(inputs=input_data_strategy())
    def test_residual_connection(self, inputs: dict[str, Any], device: str):
        """Verify residual connection works correctly."""
        # Create module and input tensors
        module = MultilevelSelfAttentionBlockWithRoPE(**inputs["config"]).to(device)
        input_tensors = make_input_tensors(**inputs["tensor_config"], device=device)

        # ensure input isn't all zeros
        x = input_tensors["x"]
        assume(not torch.allclose(x, torch.zeros_like(x)))

        # Zero out all weights to produce an identity mapping
        with torch.no_grad():
            module.qkv.weight.zero_()
            module.out_proj.weight.zero_()

            if hasattr(module.qkv, "bias") and module.qkv.bias is not None:
                module.qkv.bias.zero_()
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.zero_()

        # Make hook to get norm output
        hook = ModuleHook(module, {"norm": lambda m: m.norm})

        # Run forward with zero weights - should just get the input back due to residual
        with torch.no_grad(), hook:
            output = module(**input_tensors)

        # Output should be equal to input due to residual connection (plus normalization effect)
        if module.norm_first:
            # Input is already normalized, so output should equal input
            assert torch.allclose(
                output, input_tensors["x"]
            ), "Expected output to equal input with residual connection"
        else:
            # Output should equal norm output
            assert output.shape == input_tensors["x"].shape
            assert torch.allclose(output, hook.captured_values["norm"]["outputs"][0])

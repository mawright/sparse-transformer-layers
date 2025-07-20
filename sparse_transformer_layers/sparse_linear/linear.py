from typing import Optional

import torch
from torch import Tensor

from pytorch_sparse_utils.indexing.utils import (
    get_sparse_index_mapping,
)
from .autograd import GatherAndLinearFunction


@torch.jit.script
def batch_sparse_index_linear(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    check_all_specified: bool = False,
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor followed by a
    linear transformation. Should be equivalent to
    F.linear(sparse_tensor[index_tensor], weight, bias). The values are
    retrieved using get_sparse_index_mapping. Then, the retrieved values are
    linearly transformed according to the input weight and optional bias in
    a custom autograd function to avoid storing an extra tensor of the retrieved
    sparse values.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
            S leading sparse dimensions and M is the dense dimension.
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
            leading batch dimensions. Negative indices are not supported and will
            be considered unspecified.
        weight (Tensor): Weight matrix, of shape [out_dim, in_dim]
        bias (Optional[Tensor]): Optional bias vector, of shape [out_dim]
        check_all_specified (bool): If True, this function will raise a
            ValueError if any of the indices in `index_tensor` are not specified
            in `sparse_tensor`. If False, selections at unspecified indices will be
            returned with padding values of 0. Defaults to False.

    Returns:
        Tensor: Tensor of dimension ..., M; where the leading dimensions are
            the same as the batch dimensions from `index_tensor`.
        Tensor: Boolean tensor of dimension ...; where each element is True if
            the corresponding index is a specified (nonzero) element of the sparse
            tensor and False if not.
    """
    if index_tensor.is_nested:
        raise ValueError("Nested index tensor not supported")
        # return __gather_nested_index(sparse_tensor, index_tensor, check_all_specified)

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    transformed: Tensor = GatherAndLinearFunction.apply(
        sparse_tensor_values, index_search, is_specified_mask, weight, bias
    )  # pyright: ignore[reportAssignmentType]

    out_shape = index_tensor.shape[:-1] + (weight.size(0),)
    assert transformed.shape == out_shape
    assert is_specified_mask.shape == out_shape[:-1]

    return transformed, is_specified_mask

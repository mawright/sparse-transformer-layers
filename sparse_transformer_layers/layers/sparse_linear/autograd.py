from typing import Optional
from pytorch_sparse_utils.indexing.utils import gather_mask_and_fill


import torch
import torch.nn.functional as F
from torch import Tensor


class GatherAndLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
        is_specified_mask: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Performs F.linear(sparse_tensor_values[index_search], weight, bias)
        with minimal memory use.
        """
        ctx.set_materialize_grads(False)

        selected = gather_mask_and_fill(
            sparse_tensor_values, index_search, is_specified_mask
        )
        out = F.linear(selected, weight, bias)

        ctx.save_for_backward(
            sparse_tensor_values, weight, bias  # pyright: ignore[reportArgumentType]
        )
        ctx.index_search = index_search  # pyright: ignore[reportAttributeAccessIssue]
        ctx.is_specified_mask = (  # pyright: ignore[reportAttributeAccessIssue]
            is_specified_mask
        )

        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # pragma: no cover
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        sparse_tensor_values, weight, bias = (
            ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        )
        index_search = ctx.index_search  # pyright: ignore[reportAttributeAccessIssue]
        is_specified_mask = (
            ctx.is_specified_mask  # pyright: ignore[reportAttributeAccessIssue]
        )

        grad_values = None
        grad_weight = None
        grad_bias = None

        if grad_output is not None:
            if (
                bias is not None
                and ctx.needs_input_grad[4] # pyright: ignore[reportAttributeAccessIssue]
            ):
                grad_bias = grad_output.sum(0)

            if ctx.needs_input_grad[3]:  # pyright: ignore[reportAttributeAccessIssue]
                selected = gather_mask_and_fill(
                    sparse_tensor_values, index_search, is_specified_mask
                )
                grad_weight = torch.mm(grad_output.t(), selected)

            if ctx.needs_input_grad[0]:  # pyright: ignore[reportAttributeAccessIssue]
                grad_selected = torch.mm(grad_output, weight)
                grad_selected.masked_fill_(~is_specified_mask.unsqueeze(-1), 0)

                grad_values = torch.zeros_like(sparse_tensor_values)
                grad_values.index_add_(0, index_search, grad_selected)

        return grad_values, None, None, grad_weight, grad_bias

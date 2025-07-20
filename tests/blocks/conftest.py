from functools import wraps
from typing import Any, Callable, Dict

import torch
from torch import nn

class ModuleHook:
    """Universal hook for capturing intermediate tensors in a module."""

    def __init__(self, module: nn.Module, points_to_hook: Dict[str, Callable]):
        """
        Initialize hook.

        Args:
            module: Module to hook
            points_to_hook: Dict mapping hook names to functions that return the component to hook
        """
        self.module = module
        self.captured_values: dict[str, dict[str, Any]] = {
            hook_name: {"inputs": {}, "outputs": []} for hook_name in points_to_hook
        }
        self.original_methods = {}

        for hook_name, component_getter in points_to_hook.items():
            component = component_getter(module)
            if isinstance(component, nn.Module):
                # For submodules
                self._hook_module(hook_name, component_getter)
            else:
                # For methods
                self._hook_method(hook_name, component_getter)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def _process_value(self, value: Any) -> Any:
        """Recursively process values, detaching and cloning tensors."""
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        elif isinstance(value, (list, tuple)):
            return type(value)(self._process_value(item) for item in value)
        elif isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        return value

    def _normalize_to_list(self, value: Any) -> list[Any]:
        """Normalize input or output to a list, handling single values and tuples."""
        if isinstance(value, tuple):
            return list(self._process_value(value))
        return [self._process_value(value)]

    def _hook_method(self, hook_name: str, component_getter: Callable):
        """Hook a method to capture inputs and outputs."""
        original_method = component_getter(self.module)
        self.original_methods[hook_name] = original_method

        @wraps(original_method)
        def hooked_method(*args, **kwargs):
            # Capture inputs
            captured_args = self._process_value(args)
            captured_kwargs = self._process_value(kwargs)
            self.captured_values[hook_name]["inputs"] = {
                "args": captured_args,
                "kwargs": captured_kwargs,
            }

            # Call original method
            result = original_method(*args, **kwargs)

            # Capture outputs (normalized to list)
            self.captured_values[hook_name]["outputs"] = self._normalize_to_list(result)

            return result

        # Replace the method
        self._set_attr_nested(hook_name, hooked_method)

    def _hook_module(self, hook_name: str, component_getter: Callable):
        """Hook a module to capture inputs and outputs."""
        component: nn.Module = component_getter(self.module)

        def forward_hook(module, args, kwargs, outputs):
            captured_args = self._process_value(args)
            captured_kwargs = self._process_value(kwargs)
            self.captured_values[hook_name]["inputs"] = {
                "args": captured_args,
                "kwargs": captured_kwargs,
            }

            # Capture outputs (normalized to list)
            self.captured_values[hook_name]["outputs"] = self._normalize_to_list(
                outputs
            )

        # Register hooks
        handle = component.register_forward_hook(forward_hook, with_kwargs=True)
        self.original_methods[hook_name] = handle

    def _set_attr_nested(self, path: str, value: Any):
        """Set attribute following nested path."""
        parts = path.split(".")
        obj = self.module
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def restore(self):
        """Restore original methods and remove hooks."""
        for hook_name, original in self.original_methods.items():
            if callable(original):
                # Restore method
                self._set_attr_nested(hook_name, original)
            else:
                # Remove hook
                original.remove()

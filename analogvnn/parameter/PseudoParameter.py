from __future__ import annotations

from typing import Callable, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleDict
from torch.nn.utils import parametrize

__all__ = ['PseudoParameter']


class PseudoParameter(nn.Module):
    """A parameterized parameter which acts like a normal parameter during gradient updates.

    PyTorch's ParameterizedParameters vs AnalogVNN's PseudoParameters:

    - Similarity (Forward or Parameterizing the data):
        > Data -> ParameterizingModel -> Parameterized Data

    - Difference (Backward or Gradient Calculations):
      - ParameterizedParameters
        > Parameterized Data -> ParameterizingModel -> Data
      - PseudoParameters
        > Parameterized Data -> Data

    Attributes:
        _transformation (Callable): the transformation.
        _transformed (nn.Parameter): the transformed parameter.

    Properties:
        grad (Tensor): the gradient of the parameter.
        module (PseudoParameterModule): the module that wraps the parameter and the transformation.
        transformation (Callable): the transformation.
    """

    _transformation: Callable
    _transformed: nn.Parameter

    @staticmethod
    def identity(x: Any) -> Any:
        """The identity function.

        Args:
            x (Any): the input tensor.

        Returns:
            Any: the input tensor.
        """

        return x

    def __init__(self, data=None, requires_grad=True, transformation=None):
        """Initializes the parameter.

        Args:
            data: the data for the parameter.
            requires_grad (bool): whether the parameter requires gradient.
            transformation (Callable): the transformation.
        """

        super().__init__()
        self.original = nn.Parameter(data=data, requires_grad=requires_grad)
        self._transformed = nn.Parameter(data=data, requires_grad=requires_grad)
        self._transformed.original = self
        self._transformation = self.identity
        self.set_transformation(transformation)
        self.substitute_member(self.original, self._transformed, 'grad')

    def __call__(self, *args, **kwargs):
        """Transforms the parameter.

        Args:
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            nn.Parameter: the transformed parameter.

        Raises:
            RuntimeError: if the transformation callable fails.
        """

        try:
            self._transformed.data = self._transformation(self.original)
        except Exception as e:
            raise RuntimeError(f'here: {e.args}') from e
        return self._transformed

    def set_original_data(self, data: Tensor) -> PseudoParameter:
        """Set data to the original parameter.

        Args:
            data (Tensor): the data to set.

        Returns:
            PseudoParameter: self.
        """

        self.original.data = data
        return self

    forward = __call__
    """Alias for __call__"""

    _call_impl = __call__
    """Alias for __call__"""

    right_inverse = set_original_data
    """Alias for set_original_data."""

    def __repr__(self):
        """Returns a string representation of the parameter.

        Returns:
            str: the string representation.
        """

        return f'{self.__class__.__name__}(' \
               f'transform={self.transformation}' \
               f', original={self.original}' \
               f')'

    @property
    def transformation(self):
        """Returns the transformation.

        Returns:
            Callable: the transformation.
        """

        return self._transformation

    @transformation.setter
    def transformation(self, transformation: Callable):
        """Sets the transformation.

        Args:
            transformation (Callable): the transformation.
        """

        self.set_transformation(transformation)

    def set_transformation(self, transformation) -> PseudoParameter:
        """Sets the transformation.

        Args:
            transformation (Callable): the transformation.

        Returns:
            PseudoParameter: self.
        """

        self._transformation = transformation
        if isinstance(self._transformation, nn.Module):
            self._transformation.eval()
        return self

    @staticmethod
    def substitute_member(
            tensor_from: Any,
            tensor_to: Any,
            property_name: str,
            setter: bool = True
    ):
        """Substitutes a member of a tensor as property of another tensor.

        Args:
            tensor_from (Any): the tensor property to substitute.
            tensor_to (Any): the tensor property to substitute to.
            property_name (str): the name of the property.
            setter (bool): whether to substitute the setter.
        """

        def getter_fn(self):
            return getattr(tensor_to, property_name)

        def setter_fn(self, value):
            setattr(tensor_to, property_name, value)

        new_class = type(tensor_from.__class__.__name__, (tensor_from.__class__,), {})

        if not setter:
            setattr(new_class, property_name, property(getter_fn))
        else:
            setattr(new_class, property_name, property(getter_fn, setter_fn))

        tensor_from.__class__ = new_class

    @classmethod
    def parameterize(cls, module: nn.Module, param_name: str, transformation: Callable) -> PseudoParameter:
        """Parameterizes a parameter.

        Args:
            module (nn.Module): the module.
            param_name (str): the name of the parameter.
            transformation (Callable): the transformation to apply.

        Returns:
            PseudoParameter: the parameterized parameter.
        """

        assert hasattr(module, param_name)

        param = getattr(module, param_name)
        new_param = cls(data=param, requires_grad=param.requires_grad, transformation=transformation)

        delattr(module, param_name)

        if not parametrize.is_parametrized(module):
            # Change the class
            parametrize._inject_new_class(module)
            # Inject a ``ModuleDict`` into the instance under module.parametrizations
            module.parametrizations = ModuleDict()

        module.parametrizations[param_name] = new_param
        parametrize._inject_property(module, param_name)
        return new_param

    @classmethod
    def parametrize_module(cls, module: nn.Module, transformation: Callable, requires_grad: bool = True):
        """Parametrize all parameters of a module.

        Args:
            module (nn.Module): the module parameters to parametrize.
            transformation (Callable): the transformation.
            requires_grad (bool): if True, only parametrized parameters that require gradients.
        """

        with torch.no_grad():
            for name, parameter in list(module.named_parameters(recurse=False)):
                if isinstance(parameter, cls):
                    continue

                if requires_grad and not parameter.requires_grad:
                    continue

                cls.parameterize(module=module, param_name=name, transformation=transformation)

            for sub_module in module.children():
                if sub_module == module:
                    continue
                if hasattr(module, 'parametrizations') and (
                        sub_module is module.parametrizations or sub_module in module.parametrizations
                ):
                    continue
                cls.parametrize_module(module=sub_module, transformation=transformation, requires_grad=requires_grad)

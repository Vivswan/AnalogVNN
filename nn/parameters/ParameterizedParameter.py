from typing import Type

import torch

from nn.parameters.Parameter import Parameter
from nn.parameters.Tensor import Tensor
from nn.utils.common_types import TENSOR_CALLABLE


class Parameterization:
    @staticmethod
    def identity(x):
        return x

    def __init__(self, data=None, transform=None):
        self._transform: TENSOR_CALLABLE = Parameterization.identity if transform is None else transform
        if data is None:
            self._original = None
        else:
            self._original = torch.clone(data).detach()
            self._original.requires_grad_(data.requires_grad)

    @property
    def original(self):
        return self._original

    def set_transformation(self, transform):
        self._transform = transform
        return self

    @property
    def transformation(self):
        if hasattr(self, "_transform"):
            return self._transform
        else:
            return Parameterization.identity

    @transformation.setter
    def transformation(self, transform):
        self.set_transformation(transform)

    @property
    def transformed(self):
        with torch.no_grad():
            transformed_tensor = self.transformation(self.original)
            transformed_tensor.requires_grad_(self.original.requires_grad)
            transformed_tensor.register_hook(self.return_grad_to_original)
            transformed_tensor.grad = self.original.grad
        return transformed_tensor

    def return_grad_to_original(self, grad):
        self.original.grad = grad
        return grad

    @staticmethod
    def __static_torch_function__(cls: Type[torch.Tensor], func, types, args=(), kwargs=None):
        lambda_class = torch.Tensor
        # print("\n#### func: ", func.__name__)
        if kwargs is None:
            kwargs = {}
        args_dash = []
        for arg in args:
            if isinstance(arg, cls) and hasattr(arg, 'transformed') and hasattr(arg, 'original'):
                if func.__name__ == "__set__":
                    args_dash.append(arg.original)
                else:
                    args_dash.append(arg.transformed)
            else:
                args_dash.append(arg)

        types = [lambda_class if a == cls else a for a in types]
        ans = lambda_class.__torch_function__(func=func, types=types, args=args_dash, kwargs=kwargs)
        return ans


class ParameterizedTensor(Tensor, Parameterization):
    def __init__(self, data=None, transform=None):
        Tensor.__init__(self, data=data)
        Parameterization.__init__(self, transform=transform)

    @torch.no_grad()
    def __repr__(self):
        return f'ParameterizedTensor {self.original.__repr__()}: ' + Tensor.__repr__(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs)


class ParameterizedParameter(Parameter, Parameterization):
    def __init__(self, data=None, requires_grad=True, transform=None):
        Parameter.__init__(self, data=data, requires_grad=requires_grad)
        Parameterization.__init__(self, data=Parameter(data, requires_grad=requires_grad), transform=transform)

    @torch.no_grad()
    def __repr__(self):
        return f'ParameterizedParameter {self.original.__repr__()}: ' + Tensor.__repr__(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs)

    @staticmethod
    def sanitize_parameters_list(parameters):
        results = []
        for i in parameters:
            if isinstance(i, ParameterizedParameter):
                results.append(i.original)
            else:
                results.append(i)
        return results

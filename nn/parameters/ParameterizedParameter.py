from typing import Type

import torch
from torch import nn
from torch.nn.functional import linear
from torch.nn.utils import parametrize

from crc.cleo_run import save_graph
from nn.parameters.Parameter import Parameter
from nn.parameters.Tensor import Tensor
from nn.utils.types import TENSOR_CALLABLE


class Parameterization:
    @staticmethod
    def identity(x):
        return x

    def __init__(self, transform=None):
        self._transform: TENSOR_CALLABLE = Parameterization.identity if transform is None else transform

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
    def original(self):
        return Tensor(self).detach()

    @staticmethod
    def __static_torch_function__(cls: Type[torch.Tensor], func, types, args=(), kwargs=None, leaf=False):
        lambda_class = Tensor

        if kwargs is None:
            kwargs = {}
        args_dash = []
        for arg in args:
            if isinstance(arg, cls) and hasattr(arg, 'transformation') and func.__name__ != "__set__":
                arg = arg.transformation(lambda_class(arg))
                if leaf:
                    arg = arg.detach()
                args_dash.append(arg)
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
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs, leaf=False)


class ParameterizedParameter(Parameter, Parameterization):
    def __init__(self, data=None, requires_grad=True, transform=None):
        Parameter.__init__(self, data=data, requires_grad=requires_grad)
        Parameterization.__init__(self, transform=transform)

    @torch.no_grad()
    def __repr__(self):
        return f'ParameterizedParameter {self.original.__repr__()}: ' + Tensor.__repr__(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs, leaf=False)


if __name__ == '__main__':
    class Double(nn.Module):
        def forward(self, x):
            return x * 2


    linear1 = nn.Linear(4, 1)
    linear1.weight.data = torch.ones_like(linear1.weight.data)
    linear1.bias.data = torch.ones_like(linear1.bias.data)
    ParameterizedParameter.convert_model(linear1, transform=Double())
    ans1 = linear(torch.ones(1, 4), linear1.weight, linear1.bias)

    linear2 = nn.Linear(4, 1)
    linear2.weight.data = torch.ones_like(linear2.weight.data)
    linear2.bias.data = torch.ones_like(linear2.bias.data)
    parametrize.register_parametrization(linear2, "weight", Double())
    parametrize.register_parametrization(linear2, "bias", Double())
    ans2 = linear(torch.ones(1, 4), linear2.weight, linear2.bias)

    print(linear1.weight)
    print(linear2.weight)
    save_graph("ans1", ans1)
    save_graph("ans2", ans2)
    print(ans1 == ans2)
    print(ans1)
    print(ans2)
    print("hi")
    # n = 2
    # t = ParameterizedTensor(data=torch.eye(n), transform=nn.Flatten(start_dim=0))
    # print("get:", t)
    # print("t original:", t.original)
    # print()
    # print("add:", t + nn.Flatten(start_dim=0)(torch.eye(n)))
    # print()
    # p = ParameterizedParameter(data=torch.eye(n), requires_grad=True, transform=lambda x: x * 2)
    # print("p:", p)
    # print()
    # print("p add:", p + torch.eye(n))
    # print()
    # layer = TestModule(data=torch.eye(n))
    # print("layer.p:", layer.p)
    # print("layer p original:", layer.p.original)
    # print()
    # print("layer(1*2+1):", layer(torch.eye(n)))
    # print()
    # layer.p.data = layer.p.original * 2.5
    # print("layer(2.5*2+5):", layer(torch.eye(n) * 5))
    # print("layer p original:", layer.p.original)

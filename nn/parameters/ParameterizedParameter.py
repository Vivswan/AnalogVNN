from typing import Type

import torch

from nn.parameters.BaseParameter import BaseParameter


class Parameterization:
    @staticmethod
    def identity(x):
        return x

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._transform = Parameterization.identity

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
        lambda_class = torch.Tensor
        return lambda_class(self).detach()

    @staticmethod
    def __static_torch_function__(cls: Type[torch.Tensor], func, types, args=(), kwargs=None, leaf=False):
        lambda_class = torch.Tensor

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


class ParameterizedTensor(torch.Tensor, Parameterization):
    def __repr__(self):
        return f'ParameterizedTensor: ' + super(ParameterizedTensor, self).__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs, leaf=False)


class ParameterizedParameter(BaseParameter, Parameterization):
    def __repr__(self):
        return f'ParameterizedParameter: ' + super(ParameterizedParameter, self).__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return Parameterization.__static_torch_function__(cls, func, types, args, kwargs, leaf=True)


# class TestModule(BaseLayer):
#     def __init__(self, data):
#         super().__init__()
#         self.p = ParameterizedParameter(data=data, requires_grad=True).set_transformation(lambda x: x * 2)
#
#     def forward(self, x: Tensor) -> Tensor:
#         return x + self.p
#
#
# if __name__ == '__main__':
#     n = 2
#     t = ParameterizedTensor(torch.eye(n)).set_transformation(nn.Flatten(start_dim=0))
#     print("get:", t)
#     print("t original:", t.original)
#     print()
#     print("add:", t + nn.Flatten(start_dim=0)(torch.eye(n)))
#     print()
#     p = ParameterizedParameter(data=torch.eye(n), requires_grad=True).set_transformation(lambda x: x * 2)
#     print("p:", p)
#     print()
#     print("p add:", p + torch.eye(n))
#     print()
#     layer = TestModule(data=torch.eye(n))
#     print("layer.p:", layer.p)
#     print("layer p original:", layer.p.original)
#     print()
#     print("layer(1*2+1):", layer(torch.eye(n)))
#     print()
#     layer.p.data = layer.p.original * 2.5
#     print("layer(2.5*2+5):", layer(torch.eye(n) * 5))
#     print("layer p original:", layer.p.original)

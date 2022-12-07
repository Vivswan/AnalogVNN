from typing import Type

import torch
from torch import nn, optim, Tensor

from nn.modules.Linear import Linear
from nn.parameters.Parameter import Parameter
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


if __name__ == '__main__':
    device = "cuda"


    class Double(nn.Module):
        def forward(self, x):
            return x * 2


    # s = (2,)
    # p = ParameterizedParameter(torch.ones(*s), requires_grad=True, transform=Double())
    # a: Tensor = p * 2
    # a.backward(torch.ones(*s))
    # a: Tensor = p * 2
    # a.backward(torch.ones(*s))
    # print("p:               ", p)
    # print("p.grad:          ", p.grad)
    # print("p.original:      ", p.original)
    # print("p.original.grad: ", p.original.grad)
    # print("a:               ", a)
    # print("a.grad:          ", a.grad)

    linear1 = Linear(4, 1)
    linear1.weight.data = torch.ones_like(linear1.weight.data, requires_grad=True)
    linear1.bias.data = torch.ones_like(linear1.bias.data, requires_grad=True)
    linear1.to(device=device)
    ParameterizedParameter.convert_model(linear1, transform=Double())

    adam = optim.Adam(ParameterizedParameter.sanitize_parameters_list(linear1.parameters()), lr=1)
    print()
    print("linear1.parameters(): ", list(linear1.parameters()))
    print()
    print("linear1.parameters(): ", list(linear1.named_parameters()))
    print()

    ans1: Tensor = linear1(torch.ones(1, 4, device=device))
    # save_graph("ans1", ans1)
    ans1.backward()
    print("linear1.weight:       ", linear1.weight)
    print("linear1.weight.grad:  ", linear1.weight.grad)
    print("linear1.bias:         ", linear1.bias)
    print("linear1.bias.grad:    ", linear1.bias.grad)
    print("ans:                  ", ans1)

    print()
    adam.step()
    print("linear1.weight:       ", linear1.weight)
    print("linear1.weight.grad:  ", linear1.weight.grad)
    print("linear1.bias:         ", linear1.bias)
    print("linear1.bias.grad:    ", linear1.bias.grad)

    print()
    adam.zero_grad()
    print("linear1.weight:       ", linear1.weight)
    print("linear1.weight.grad:  ", linear1.weight.grad)
    print("linear1.bias:         ", linear1.bias)
    print("linear1.bias.grad:    ", linear1.bias.grad)

    print()
    print("linear1.parameters(): ", list(linear1.parameters()))
    print()
    print("linear1.parameters(): ", list(linear1.named_parameters()))
    print()

    # linear2 = nn.Linear(4, 1)
    # linear2.weight.data = torch.ones_like(linear2.weight.data)
    # linear2.bias.data = torch.ones_like(linear2.bias.data)
    # parametrize.register_parametrization(linear2, "weight", Double())
    # parametrize.register_parametrization(linear2, "bias", Double())
    # ans2 = linear2(torch.ones(1, 4))

    # print(linear1.weight)
    # print(linear1.weight.grad)
    # print(linear2.weight)
    # print()
    # save_graph("ans2", ans2)
    # print()
    # print(ans1 == ans2)
    # print(ans1)
    # print(ans2)
    # print("hi")
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

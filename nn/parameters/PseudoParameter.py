import torch
from torch import nn, optim

from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.optimizer.BaseOptimizer import set_grad_zero
from nn.parameters.Parameter import Parameter
from nn.parameters.Tensor import Tensor
from nn.utils.common_types import TENSOR_CALLABLE


class PseudoParameter(Parameter):
    @staticmethod
    def identity(x):
        return x

    def __init__(self, data=None, requires_grad=True, transform=None, initialise_zero_pseudo=False):
        super(PseudoParameter, self).__init__(data, requires_grad)

        self._original = None
        self.grad_hook = None
        self.initialise_zero_pseudo = initialise_zero_pseudo
        self._transform: TENSOR_CALLABLE = PseudoParameter.identity if transform is None else transform
        self._initialise(data)

    def _initialise(self, data):
        if self.initialise_zero_pseudo:
            self.original = torch.zeros_like(data, requires_grad=False)
        else:
            self.original = torch.clone(data)
            self.original.detach_()

        self.original.parent = self
        self.original.requires_grad_(self.requires_grad)
        self.grad_hook = self.register_hook(self.return_grad_to_original)
        self.update()
        return self

    def __repr__(self):
        return f'PseudoParameter(' \
               f'initialise_zero_pseudo:{self.initialise_zero_pseudo}' \
               f', transform:{self.transformation}' \
               f', original:{self.original}' \
               f'): ' \
               + super(PseudoParameter, self).__repr__().replace("\n", " ")

    @property
    def original(self):
        return self._original

    @original.setter
    def original(self, data):
        self._original = data

    @property
    def transformation(self):
        if hasattr(self, "_transform"):
            return self._transform
        else:
            return self.identity

    def set_transformation(self, transform):
        self._transform = transform
        return self

    @transformation.setter
    def transformation(self, transform):
        self.set_transformation(transform)

    def return_grad_to_original(self, grad):
        self.original.grad = grad
        self.grad = grad
        return grad

    def zero_grad(self, set_to_none: bool = False):
        set_grad_zero(self, set_to_none=set_to_none)
        set_grad_zero(self.original, set_to_none=set_to_none)

    @torch.no_grad()
    def set_data(self, data: Tensor):
        data.requires_grad_(False)
        self.data = self.transformation(data)
        self.grad = self.original.grad
        return self

    @torch.no_grad()
    def update(self):
        return self.set_data(self.original)

    @staticmethod
    def sanitize_parameters(parameters):
        if isinstance(parameters, nn.Module):
            parameters = parameters.parameters()

        results = []
        for i in parameters:
            if isinstance(i, PseudoParameter):
                results.append(i.original)
            else:
                results.append(i)
        return results

    @staticmethod
    def update_params(parameters):
        if isinstance(parameters, nn.Module):
            parameters = parameters.parameters()

        for i in parameters:
            if isinstance(i, PseudoParameter):
                i.update()


if __name__ == '__main__':
    class Double(nn.Module):
        def forward(self, x):
            return x * 2


    linear1 = nn.Linear(4, 1)
    linear1.weight.data = torch.ones_like(linear1.weight.data, requires_grad=True)
    linear1.bias.data = torch.ones_like(linear1.bias.data, requires_grad=True)
    PseudoParameter.convert_model(linear1, transform=GaussianNoise(std=1))

    adam = optim.Adam(PseudoParameter.sanitize_parameters(linear1), lr=1)
    print()
    print("linear1.parameters():          ", list(linear1.parameters()))
    print("linear1.parameters():          ", list(linear1.named_parameters()))
    print()

    ans1: Tensor = linear1(torch.ones(1, 4))
    # save_graph("ans1", ans1)
    ans1.backward()
    print("linear1.weight:                ", linear1.weight)
    print("linear1.weight.grad:           ", linear1.weight.grad)
    print("linear1.weight.original:       ", linear1.weight.original)
    print("linear1.weight.original.grad:  ", linear1.weight.original.grad)
    print("linear1.bias:                  ", linear1.bias)
    print("linear1.bias.grad:             ", linear1.bias.grad)
    print("linear1.bias.original:         ", linear1.bias.original)
    print("linear1.bias.original.grad:    ", linear1.bias.original.grad)
    print("ans:                           ", ans1)

    print()
    adam.step()
    print("############ step ############")
    print("linear1.weight:                ", linear1.weight)
    print("linear1.weight.grad:           ", linear1.weight.grad)
    print("linear1.weight.original:       ", linear1.weight.original)
    print("linear1.weight.original.grad:  ", linear1.weight.original.grad)
    print("linear1.bias:                  ", linear1.bias)
    print("linear1.bias.grad:             ", linear1.bias.grad)
    print("linear1.bias.original:         ", linear1.bias.original)
    print("linear1.bias.original.grad:    ", linear1.bias.original.grad)

    print()
    adam.zero_grad()
    print("########### zero_grad ###########")
    print("linear1.weight:                ", linear1.weight)
    print("linear1.weight.grad:           ", linear1.weight.grad)
    print("linear1.weight.original:       ", linear1.weight.original)
    print("linear1.weight.original.grad:  ", linear1.weight.original.grad)
    print("linear1.bias:                  ", linear1.bias)
    print("linear1.bias.grad:             ", linear1.bias.grad)
    print("linear1.bias.original:         ", linear1.bias.original)
    print("linear1.bias.original.grad:    ", linear1.bias.original.grad)

    print()
    print("linear1.parameters():          ", list(linear1.parameters()))
    print("linear1.parameters():          ", list(linear1.named_parameters()))
    print()

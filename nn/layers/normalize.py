from abc import ABC

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer
from nn.utils.is_using_cuda import get_device


class Normalize(BaseLayer, BackwardFunction, ABC):
    def activation(self):
        pass


class Norm(Normalize):
    def forward(self, x: Tensor):
        if self.training:
            self.save_tensor("input", x)
            norm = x.norm()
            if torch.isclose(norm, torch.tensor(0., device=get_device())):
                return x
            else:
                return x / norm
        else:
            return x

    def backward(self, grad_output):
        x = self.get_tensor("input")
        return grad_output * x.norm()


class Clamp(Normalize):
    def forward(self, x: Tensor):
        if self.training:
            y = torch.clamp(x, min=-1, max=1)
            self.save_tensor("input", x)
            self.save_tensor("output", y)
            return y
        else:
            return x

    def backward(self, grad_output):
        x = self.get_tensor("input")
        y = self.get_tensor("output")
        return grad_output * torch.nan_to_num(x / y, nan=1)

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class Normalize(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor):
        if self.training:
            self.save_tensor("input", x)
            norm = x.norm()
            if norm == 0:
                return x
            else:
                return x / x.norm()
        else:
            return x

    def backward(self, grad_output):
        x = self.get_tensor("input")
        return grad_output * x.norm()


class Clamp(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor):
        if self.training:
            self.save_tensor("input", x)
            return torch.clamp(x, min=-1, max=1)
        else:
            return x

    def backward(self, grad_output):
        x = self.get_tensor("input")
        return grad_output

from abc import ABC
from typing import Union

import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardIdentity
from nn.layers.BaseLayer import BaseLayer


class Normalize(BaseLayer, BackwardIdentity, ABC):
    pass


class L1Norm(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        norm = x
        if len(x.shape) > 1:
            norm = torch.flatten(norm, start_dim=1)
        norm = torch.norm(norm, 1, -1)
        norm = torch.clamp(norm, min=1e-4)
        self.save_tensor("norm", norm)
        x = torch.div(x.T, norm).T
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        norm = self.get_tensor("norm")
        grad = (torch.div(torch.ones_like(x, device=grad_output.device).T, norm)).T - (
            torch.div(2 * torch.pow(x, 2).T, torch.pow(norm, 2))).T
        return grad_output * grad


class L2Norm(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        norm = x
        if len(x.shape) > 1:
            norm = torch.flatten(norm, start_dim=1)
        norm = torch.norm(norm, 2, -1)
        norm = torch.clamp(norm, min=1e-4)
        self.save_tensor("norm", norm)

        x = torch.div(x.T, norm).T
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        norm = self.get_tensor("norm")
        grad = (torch.div(torch.ones_like(x, device=grad_output.device).T, norm)).T - (
            torch.div(torch.pow(x, 2).T, torch.pow(norm, 2.5))).T
        return grad_output * grad


class L1NormW(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        norm = torch.norm(x, 1)
        norm = torch.clamp(norm, min=1e-4)
        x = torch.div(x, norm)
        return x


class L2NormW(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        norm = torch.norm(x, 2)
        norm = torch.clamp(norm, min=1e-4)
        x = torch.div(x, norm)
        return x


class Clamp(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        return torch.clamp(x, min=-1, max=1)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = ((-1 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad


class Clamp01(Normalize):
    def forward(self, x: Tensor):
        self.save_tensor("input", x)
        return torch.clamp(x, min=0, max=1)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = ((0 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad

import inspect
from typing import Type

import torch
from torch.optim import Optimizer


def set_grad_zero(tensor, set_to_none):
    if tensor.grad is not None:
        if set_to_none:
            tensor.grad = None
        else:
            if tensor.grad.grad_fn is not None:
                tensor.grad.detach_()
            else:
                tensor.grad.requires_grad_(False)
            tensor.grad.zero_()


class BaseOptimizer(Optimizer):
    def __init__(self, optimizer_cls: Type[Optimizer], params, defaults=None, **kwargs):
        self.optimizer_cls = optimizer_cls

        if defaults is None:
            defaults = dict()

        optimizer_signature = inspect.signature(optimizer_cls)

        for parameter, value in optimizer_signature.parameters.items():
            defaults[parameter] = None if value.default == optimizer_signature.empty else value.default

        for parameter, value in kwargs.items():
            defaults[parameter] = value

        defaults["optimizer_cls"] = optimizer_cls
        super(BaseOptimizer, self).__init__(params, defaults)

    @staticmethod
    def add_optimizer_to_param_group(param_group: dict) -> None:
        class_parameters = {}
        for parameter, value in inspect.signature(param_group['optimizer_cls']).parameters.items():
            class_parameters[parameter] = param_group[parameter]
        param_group['optimizer'] = param_group['optimizer_cls'](**class_parameters)

    @torch.no_grad()
    def step(self, closure=None, set_to_none: bool = False):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'optimizer' not in group:
                self.add_optimizer_to_param_group(group)

            group['optimizer'].step()

        return loss

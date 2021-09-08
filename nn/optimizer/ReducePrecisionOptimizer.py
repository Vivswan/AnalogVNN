import inspect
from typing import Type

import torch
from torch.optim import Optimizer


class ReducePrecisionOptimizer(Optimizer):

    def __init__(self, optimizer_cls: Type[Optimizer], params, **kwargs):
        self.optimizer_cls = optimizer_cls

        defaults = dict()
        optimizer_signature = inspect.signature(optimizer_cls)
        for parameter, value in optimizer_signature.parameters.items():
            defaults[parameter] = None if value.default == optimizer_signature.empty else value.default
        for parameter, value in kwargs.items():
            defaults[parameter] = value
        defaults["optimizer_cls"] = optimizer_cls

        super(ReducePrecisionOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

        return loss

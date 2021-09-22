import inspect
from typing import Type

import torch
from torch.optim import Optimizer

from nn.optimizer.BaseOptimizer import BaseOptimizer, set_grad_zero
from nn.parameters.BasePrecisionParameter import BasePrecisionParameter


class BasePrecisionOptimizer(BaseOptimizer):
    def __init__(self, optimizer_cls: Type[Optimizer], parameter_class: Type[BasePrecisionParameter], params,
                 defaults=None, **kwargs):
        if defaults is None:
            defaults = dict()

        defaults['parameter_class'] = parameter_class
        super(BasePrecisionOptimizer, self).__init__(optimizer_cls, params, defaults, **kwargs)

    @torch.no_grad()
    def step(self, closure=None, set_to_none: bool = False):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'optimizer' not in group:
                parameter_for_optimizer = []

                for p in group['params']:
                    if isinstance(p, group['parameter_class']):
                        parameter_for_optimizer.append(p.pseudo_tensor)
                    else:
                        parameter_for_optimizer.append(p)

                class_parameters = {}
                for parameter, value in inspect.signature(group['optimizer_cls']).parameters.items():
                    class_parameters[parameter] = group[parameter]
                class_parameters["params"] = parameter_for_optimizer
                group['optimizer'] = group['optimizer_cls'](**class_parameters)

            rp_parameter_with_grad = []
            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, group['parameter_class']):
                    if p.pseudo_tensor.grad is None:
                        p.pseudo_tensor.grad = torch.clone(p.grad)
                        p.pseudo_tensor.grad.detach_()
                        p.pseudo_tensor.grad.requires_grad_(False)
                    else:
                        p.pseudo_tensor.grad += p.grad
                    set_grad_zero(p, set_to_none=set_to_none)
                    rp_parameter_with_grad.append(p.pseudo_tensor)

            group['optimizer'].step()

            for p in group['params']:
                if isinstance(p, group['parameter_class']):
                    self.step_precision_parameter(p, group)

        return loss

    @torch.no_grad()
    def step_precision_parameter(self, parameter, group) -> Type[BasePrecisionParameter]:
        raise NotImplementedError

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        super(BasePrecisionOptimizer, self).zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                if isinstance(p, group['parameter_class']):
                    set_grad_zero(p.pseudo_tensor, set_to_none)

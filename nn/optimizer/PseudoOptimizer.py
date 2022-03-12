import inspect
from typing import Type

import torch
from torch.optim import Optimizer

from nn.optimizer.BaseOptimizer import BaseOptimizer
from nn.parameters.PseudoParameter import PseudoParameter


class PseudoOptimizer(BaseOptimizer):
    parameter_type = PseudoParameter

    def __init__(
            self,
            optimizer_cls: Type[Optimizer],
            params,
            parameter_class: Type[PseudoParameter] = None,
            defaults=None,
            **kwargs
    ):
        if defaults is None:
            defaults = dict()

        defaults['parameter_class'] = parameter_class if parameter_class is not None else self.parameter_type
        super(PseudoOptimizer, self).__init__(optimizer_cls, params, defaults, **kwargs)

    @staticmethod
    def add_optimizer_to_param_group(param_group: dict) -> None:
        class_parameters = {}
        for parameter, value in inspect.signature(param_group['optimizer_cls']).parameters.items():
            class_parameters[parameter] = param_group[parameter]
        class_parameters["params"] = param_group['parameter_class'].sanitize_parameters_list(param_group['params'])
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

            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, group['parameter_class']):
                    if p.original.grad is None:
                        p.original.grad = p.grad
                    else:
                        p.original.grad += p.grad

            group['optimizer'].step()

            for p in group['params']:
                if isinstance(p, group['parameter_class']):
                    self.step_precision_parameter(p, group)

        return loss

    @torch.no_grad()
    def step_precision_parameter(self, parameter: PseudoParameter, group) -> PseudoParameter:
        return parameter.update()

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        super(PseudoOptimizer, self).zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, "zero_grad"):
                    p.zero_grad(set_to_none)

from typing import Type

import torch
from torch.optim import Optimizer

from nn.optimizer.BasePrecisionOptimizer import BasePrecisionOptimizer
from nn.parameters.BasePrecisionParameter import BasePrecisionParameter
from nn.parameters.StochasticReducePrecisionParameter import StochasticReducePrecisionParameter


class StochasticReducePrecisionOptimizer(BasePrecisionOptimizer):
    def __init__(self, optimizer_cls: Type[Optimizer], params, **kwargs):
        super().__init__(
            optimizer_cls=optimizer_cls,
            parameter_class=StochasticReducePrecisionParameter,
            params=params,
            defaults=dict(),
            **kwargs
        )

    @torch.no_grad()
    def step_precision_parameter(self, parameter, group) -> Type[BasePrecisionParameter]:
        parameter.set_tensor(parameter.pseudo_tensor)
        return parameter

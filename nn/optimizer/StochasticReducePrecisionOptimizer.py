from typing import Type

from torch.optim import Optimizer

from nn.optimizer.PseudoOptimizer import PseudoOptimizer
from nn.parameters.StochasticReducePrecisionParameter import StochasticReducePrecisionParameter


class StochasticReducePrecisionOptimizer(PseudoOptimizer):
    def __init__(self, optimizer_cls: Type[Optimizer], params, **kwargs):
        super().__init__(
            optimizer_cls=optimizer_cls,
            parameter_class=StochasticReducePrecisionParameter,
            params=params,
            defaults=dict(),
            **kwargs
        )

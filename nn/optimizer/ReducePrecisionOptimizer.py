from enum import Enum
from typing import Type

import torch
from torch.optim import Optimizer

from nn.optimizer.PseudoOptimizer import PseudoOptimizer
from nn.parameters.ReducePrecisionParameter import ReducePrecisionParameter


class PrecisionUpdateTypes(Enum):
    WEIGHT_UPDATE = "WEIGHT_UPDATE"
    FULL_WEIGHT_UPDATE = "FULL_WEIGHT_UPDATE"
    THRESHOLD_WEIGHT_UPDATE = "THRESHOLD_WEIGHT_UPDATE"
    THRESHOLD_FULL_WEIGHT_UPDATE = "THRESHOLD_FULL_WEIGHT_UPDATE"


class ReducePrecisionOptimizer(PseudoOptimizer):
    def __init__(self, optimizer_cls: Type[Optimizer], params, weight_update_type: PrecisionUpdateTypes, **kwargs):
        defaults = dict()
        if weight_update_type not in PrecisionUpdateTypes:
            raise Exception(f"invalid value for 'weight_update_type': {weight_update_type}")

        defaults["weight_update_type"] = weight_update_type

        super().__init__(
            optimizer_cls=optimizer_cls,
            parameter_class=ReducePrecisionParameter,
            params=params,
            defaults=defaults,
            **kwargs
        )

    @torch.no_grad()
    def step_precision_parameter(self, parameter: ReducePrecisionParameter, group):
        if group["weight_update_type"] == PrecisionUpdateTypes.WEIGHT_UPDATE:
            parameter.update()
            return parameter

        if group["weight_update_type"] == PrecisionUpdateTypes.FULL_WEIGHT_UPDATE:
            parameter.set_data(parameter + (torch.sign(parameter.original) * (1 / parameter.precision)))
            parameter.original.zero_()
            return parameter

        if torch.any(torch.abs(parameter.original) > 1 / parameter.precision):  # TODO
            if group["weight_update_type"] == PrecisionUpdateTypes.THRESHOLD_WEIGHT_UPDATE:
                parameter.set_data(parameter + parameter.original)
                parameter.original.zero_()
                return parameter

            if group["weight_update_type"] == PrecisionUpdateTypes.THRESHOLD_FULL_WEIGHT_UPDATE:
                parameter.set_data(parameter + (torch.sign(parameter.original) * (1 / parameter.precision)))
                parameter.original.zero_()
                return parameter

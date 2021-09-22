import inspect
from typing import Type

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

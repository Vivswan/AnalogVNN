import torch
from torch import nn


class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, *args, **kwargs):
        return super(Parameter, cls).__new__(cls, data, requires_grad)

    # noinspection PyUnusedLocal
    def __init__(self, data=None, requires_grad=True, *args, **kwargs):
        super(Parameter, self).__init__()

    def __repr__(self):
        return super(Parameter, self).__repr__()

    @classmethod
    def from_tensor(cls, tensor, *args, **kwargs):
        return cls(data=tensor, requires_grad=tensor.requires_grad, *args, **kwargs)

    @classmethod
    def from_parameter(cls, parameter, *args, **kwargs):
        return cls(data=parameter, requires_grad=parameter.requires_grad, *args, **kwargs)

    @classmethod
    def convert_model(cls, model: nn.Module, *args, **kwargs):
        with torch.no_grad():
            if len(list(model.parameters())) != len(list(model.named_parameters())):
                raise Exception("All parameters need to be named to be converted.")

            for name, parameter in model.named_parameters(recurse=False):
                if isinstance(parameter, cls):
                    continue

                if not parameter.requires_grad:
                    continue

                setattr(
                    model,
                    name,
                    cls.from_parameter(parameter=parameter, *args, **kwargs)
                )

            for module in model.children():
                if module == model:
                    continue
                cls.convert_model(module, *args, **kwargs)

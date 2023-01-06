from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.nn import ModuleDict
from torch.nn.utils import parametrize

from analogvnn.parameter.Parameter import Parameter


class PseudoParameterList(nn.Module):
    original: PseudoParameter
    _transformed: nn.Parameter

    def __init__(self, original, transformed):
        super().__init__()
        self.original = original
        self._transformed = transformed

    def _call_impl(self, *args, **kwargs):
        return self.original()

    __call__ = _call_impl
    forward = _call_impl

    def right_inverse(self, data):
        self.original.data = data


class PseudoParameter(Parameter):
    _transformation: Callable
    _transformed: nn.Parameter
    _module: PseudoParameterList

    @staticmethod
    def identity(x):
        return x

    def __init__(self, data=None, requires_grad=True, transformation=None, *args, **kwargs):
        super().__init__(data, requires_grad, *args, **kwargs)
        self._transformed = nn.Parameter(data=data, requires_grad=requires_grad)
        self._transformed.original = self
        self._transformation = self.identity
        self.set_transformation(transformation)

        self._module = PseudoParameterList(
            original=self,
            transformed=self._transformed
        )

    def __call__(self, *args, **kwargs):
        try:
            self._transformed.data = self._transformation(self)
        except Exception as e:
            raise Exception(f"here: {e.args}") from e
        return self._transformed

    def __repr__(self):
        return f'{PseudoParameter.__name__}(' \
               f'transform={self.transformation}' \
               f', data={self.data}' \
               f')'

    @property
    def grad(self):
        return self._transformed.grad

    @property
    def module(self):
        return self._module

    @property
    def transformation(self):
        return self._transformation

    def set_transformation(self, transformation):
        self._transformation = transformation
        if isinstance(self._transformation, nn.Module):
            self._transformation.eval()
        return self

    @transformation.setter
    def transformation(self, transformation):
        self.set_transformation(transformation)

    @classmethod
    def parameterize(cls, module, param_name, transformation):
        assert hasattr(module, param_name)

        param = getattr(module, param_name)
        new_param = cls(data=param, requires_grad=param.requires_grad, transformation=transformation)

        delattr(module, param_name)

        if not parametrize.is_parametrized(module):
            # Change the class
            parametrize._inject_new_class(module)
            # Inject a ``ModuleDict`` into the instance under module.parametrizations
            module.parametrizations = ModuleDict()

        module.parametrizations[param_name] = new_param.module
        parametrize._inject_property(module, param_name)
        return new_param

    @classmethod
    def parametrize_module(cls, module, transformation, requires_grad=True):
        with torch.no_grad():
            for name, parameter in list(module.named_parameters(recurse=False)):
                if isinstance(parameter, cls):
                    continue

                if requires_grad and not parameter.requires_grad:
                    continue

                cls.parameterize(module=module, param_name=name, transformation=transformation)

            for sub_module in module.children():
                if sub_module == module:
                    continue
                if hasattr(module, "parametrizations") and (
                        sub_module is module.parametrizations or sub_module in module.parametrizations
                ):
                    continue
                cls.parametrize_module(module=sub_module, transformation=transformation, requires_grad=requires_grad)

    # def __getattribute__(self, item):
    #     print(f"__getattribute__:: {item!r}")
    #     return super(PseudoParameter, self).__getattribute__(item)
    #
    # def __setattr__(self, key, value):
    #     print(f"__setattr__:: {key!r} -> {value!r}")
    #     super(PseudoParameter, self).__setattr__(key, value)
    #
    # # def __set__(self, instance, value):
    # #     print(f"__set__:: {instance!r} -> {value!r}")
    # #     super(PseudoParameter, self).__set__(instance, value)
    #
    # def __get__(self, instance, owner):
    #     print(f"__get__:: {instance!r} -> {owner!r}")
    #     return super(PseudoParameter, self).__get__(instance, owner)
    #
    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     pargs = [x for x in args if not isinstance(x, PseudoParameter)]
    #     print(f"__torch_function__:: {func}, types: {types!r}, args: {pargs!r}, kwargs:{kwargs!r}")
    #     return super().__torch_function__(func, types, args, {} if kwargs is None else kwargs)


if __name__ == '__main__':
    import torch.nn as nn
    from torch import Tensor
    from torch.optim import Adam

    from analogvnn.backward.BackwardIdentity import BackwardIdentity
    from analogvnn.nn.module.Model import Model
    from analogvnn.utils.make_dot import make_dot


    class Layer(nn.Module):
        def __init__(self):
            super().__init__()

            self.weight = nn.Parameter(
                data=torch.ones((1, 1)) * 2,
                requires_grad=True
            )

        def forward(self, x):
            return x + (torch.ones_like(x) * self.weight)


    class Symmetric(BackwardIdentity, Model):
        def forward(self, x):
            return torch.rand((1, x.size()[0])) @ x @ torch.rand((x.size()[1], 1))


    def pstr(s):
        return str(s).replace("  ", "").replace("\n", "")


    model = Layer()
    parametrization = Symmetric()
    # parametrization.eval()

    # # Set the parametrization mechanism
    # # Fetch the original buffer or parameter
    # # We create this early to check for possible errors
    # parametrizations = parametrize.ParametrizationList([parametrization], model.weight)
    # # Delete the previous parameter or buffer
    # delattr(model, "weight")
    # # If this is the first parametrization registered on the module,
    # # we prepare the module to inject the property
    # if not parametrize.is_parametrized(model):
    #     # Change the class
    #     _inject_new_class(model)
    #     # Inject a ``ModuleDict`` into the instance under module.parametrizations
    #     model.parametrizations = ModuleDict()
    # # Add a property into the class
    # _inject_property(model, "weight")
    # # Add a ParametrizationList
    # model.parametrizations["weight"] = parametrizations

    # parametrize.register_parametrization(model, "weight", parametrization)

    PseudoParameter.parameterize(model, "weight", parametrization)
    print(f"module.weight                           = {pstr(model.weight)}")
    print(f"module.weight                           = {pstr(model.weight)}")
    model.weight = torch.ones((1, 1)) * 3
    model.weight.requires_grad = False
    print(f"module.weight                           = {pstr(model.weight)}")
    model.weight.requires_grad = True
    print(f"module.weight.original                  = {pstr(model.weight.original)}")
    print(f"type(module.weight)                     = {type(model.weight)}")
    print(f"module.parameters()                     = {pstr(list(model.parameters()))}")
    print(f"module.named_parameters()               = {pstr(list(model.named_parameters(recurse=False)))}")
    print(f"module.named_parameters(recurse=True)   = {pstr(list(model.named_parameters(recurse=True)))}")
    inputs = torch.ones((2, 2), dtype=torch.float, requires_grad=True)
    output: Tensor = model(inputs)
    print(f"inputs                                  = {pstr(inputs)}")
    print(f"output                                  = {pstr(output)}")

    make_dot(output, params={
        "inputs": inputs,
        "output": output,
        "model.weight": model.weight,
        # "model.parametrizations.weight.original": model.parametrizations.weight.original,
    }).render("C:/X/_data/model_graph", format="svg", cleanup=True)

    print()
    print("Forward::")
    output: Tensor = model(inputs)
    print("Backward::")
    output.backward(gradient=torch.ones_like(output))
    print("Accessing::")
    print(f"module.weight                           = {pstr(model.weight)}")
    print(f"module.weight.original                  = {pstr(model.weight.original)}")
    print(f"module.weight.grad                      = {pstr(model.weight.grad)}")
    print(f"module.weight.original.grad             = {pstr(model.weight.original.grad)}")
    print("Update::")
    opt = Adam(params=model.parameters())
    print(f"module.weight                           = {pstr(model.weight)}")
    print(f"module.weight.original                  = {pstr(model.weight.original)}")
    print(f"module.weight.grad                      = {pstr(model.weight.grad)}")
    print(f"module.weight.original.grad             = {pstr(model.weight.original.grad)}")
    print("Step::")
    opt.step()
    print(f"module.weight                           = {pstr(model.weight)}")
    print(f"module.weight.original                  = {pstr(model.weight.original)}")
    print(f"module.weight.grad                      = {pstr(model.weight.grad)}")
    print(f"module.weight.original.grad             = {pstr(model.weight.original.grad)}")
    print("zero_grad::")
    opt.zero_grad()
    print(f"module.weight                           = {pstr(model.weight)}")
    print(f"module.weight.original                  = {pstr(model.weight.original)}")
    print(f"module.weight.grad                      = {pstr(model.weight.grad)}")
    print(f"module.weight.original.grad             = {pstr(model.weight.original.grad)}")

from __future__ import annotations

from typing import Callable, Any

import torch
import torch.nn as nn
from torch.nn import ModuleDict
from torch.nn.utils import parametrize

from analogvnn.parameter.Parameter import Parameter

__all__ = ['PseudoParameter']


class PseudoParameterModule(nn.Module):
    """A module that wraps a parameter and a function to transform it.

    Attributes:
        original (PseudoParameter): the original parameters.
        _transformed (nn.Parameter): the transformed parameters.
    """
    original: PseudoParameter
    _transformed: nn.Parameter

    def __init__(self, original, transformed):
        """Creates a new pseudo parameter module.

        Args:
            original (PseudoParameter): the original parameters.
            transformed (nn.Parameter): the transformed parameters.
        """
        super().__init__()
        self.original = original
        self._transformed = transformed

    def __call__(self, *args, **kwargs) -> nn.Parameter:
        """Transforms the parameter by calling the __call__ method of the PseudoParameter.

        Args:
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            nn.Parameter: The transformed parameter.
        """
        return self.original()

    forward = __call__
    """Alias for __call__"""

    _call_impl = __call__
    """Alias for __call__"""

    def set_original_data(self, data: Tensor) -> PseudoParameterModule:
        """set data to the original parameter.

        Args:
            data (Tensor): the data to set.

        Returns:
            PseudoParameterModule: self.
        """
        self.original.data = data
        return self

    right_inverse = set_original_data
    """Alias for set_original_data."""


class PseudoParameter(Parameter):
    """A parameterized parameter which acts like a normal parameter during gradient updates.

    PyTorch's ParameterizedParameters vs AnalogVNN's PseudoParameters:

    - Similarity (Forward or Parameterizing the data):
        > Data -> ParameterizingModel -> Parameterized Data

    - Difference (Backward or Gradient Calculations):
      - ParameterizedParameters
        > Parameterized Data -> ParameterizingModel -> Data
      - PseudoParameters
        > Parameterized Data -> Data

    Attributes:
        _transformation (Callable): the transformation.
        _transformed (nn.Parameter): the transformed parameter.
        _module (PseudoParameterModule): the module that wraps the parameter and the transformation.

    Properties:
        grad (Tensor): the gradient of the parameter.
        module (PseudoParameterModule): the module that wraps the parameter and the transformation.
        transformation (Callable): the transformation.
    """
    _transformation: Callable
    _transformed: nn.Parameter
    _module: PseudoParameterModule

    @staticmethod
    def identity(x: Any) -> Any:
        """The identity function.

        Args:
            x (Any): the input tensor.

        Returns:
            Any: the input tensor.
        """
        return x

    def __init__(self, data=None, requires_grad=True, transformation=None, *args, **kwargs):
        """Initializes the parameter.

        Args:
            data: the data for the parameter.
            requires_grad (bool): whether the parameter requires gradient.
            transformation (Callable): the transformation.
            *args: additional arguments.
            **kwargs: additional keyword arguments.
        """
        super().__init__(data, requires_grad, *args, **kwargs)
        self._transformed = nn.Parameter(data=data, requires_grad=requires_grad)
        self._transformed.original = self
        self._transformation = self.identity
        self.set_transformation(transformation)

        self._module = PseudoParameterModule(
            original=self,
            transformed=self._transformed
        )

    def __call__(self, *args, **kwargs):
        """Transforms the parameter.

        Args:
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            nn.Parameter: the transformed parameter.
        """
        try:
            self._transformed.data = self._transformation(self)
        except Exception as e:
            raise Exception(f"here: {e.args}") from e
        return self._transformed

    def __repr__(self):
        """Returns a string representation of the parameter.

        Returns:
            str: the string representation.
        """
        return f'{PseudoParameter.__name__}(' \
               f'transform={self.transformation}' \
               f', data={self.data}' \
               f')'

    @property
    def grad(self):
        """Returns the gradient of the parameter.

        Returns:
            Tensor: the gradient.
        """
        return self._transformed.grad

    @property
    def module(self):
        """Returns the module.

        Returns:
            PseudoParameterModule: the module.
        """
        return self._module

    @property
    def transformation(self):
        """Returns the transformation.

        Returns:
            Callable: the transformation.
        """
        return self._transformation

    @transformation.setter
    def transformation(self, transformation: Callable):
        """Sets the transformation.

        Args:
            transformation (Callable): the transformation.
        """
        self.set_transformation(transformation)

    def set_transformation(self, transformation) -> PseudoParameter:
        """Sets the transformation.

        Args:
            transformation (Callable): the transformation.

        Returns:
            PseudoParameter: self.
        """
        self._transformation = transformation
        if isinstance(self._transformation, nn.Module):
            self._transformation.eval()
        return self

    @classmethod
    def parameterize(cls, module: nn.Module, param_name: str, transformation: Callable) -> PseudoParameter:
        """Parameterizes a parameter.

        Args:
            module (nn.Module): the module.
            param_name (str): the name of the parameter.
            transformation (Callable): the transformation to apply.

        Returns:
            PseudoParameter: the parameterized parameter.
        """
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
    def parametrize_module(cls, module: nn.Module, transformation: Callable, requires_grad: bool = True):
        """Parametrize all parameters of a module.

        Args:
            module (nn.Module): the module parameters to parametrize.
            transformation (Callable): the transformation.
            requires_grad (bool): if True, only parametrized parameters that require gradients.
        """
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

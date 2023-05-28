import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.module.Model import Model
from analogvnn.utils.render_autograd_graph import save_autograd_graph_from_module

# def __getattribute__(self, item):
#     print(f"__getattribute__:: {item!r}")
#     return super().__getattribute__(item)
#
# def __setattr__(self, key, value):
#     print(f"__setattr__:: {key!r} -> {value!r}")
#     super().__setattr__(key, value)
#
# # def __set__(self, instance, value):
# #     print(f"__set__:: {instance!r} -> {value!r}")
# #     super().__set__(instance, value)
#
# def __get__(self, instance, owner):
#     print(f"__get__:: {instance!r} -> {owner!r}")
#     return super().__get__(instance, owner)
#
# @classmethod
# def __torch_function__(cls, func, types, args=(), kwargs=None):
#     pargs = [x for x in args if not isinstance(x, PseudoParameter)]
#     print(f"__torch_function__:: {func}, types: {types!r}, args: {pargs!r}, kwargs:{kwargs!r}")
#     return super().__torch_function__(func, types, args, {} if kwargs is None else kwargs)


if __name__ == '__main__':
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

    save_autograd_graph_from_module(output, params={
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

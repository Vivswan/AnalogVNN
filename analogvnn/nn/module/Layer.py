from __future__ import annotations

import functools
from typing import Union, Type, Callable, Sequence, Optional, TYPE_CHECKING

from torch import nn, Tensor

from analogvnn.backward.BackwardFunction import BackwardFunction
from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.graph.ArgsKwargs import ArgsKwargs

if TYPE_CHECKING:
    from analogvnn.graph.ModelGraph import ModelGraph


# https://github.com/pytorch/pytorch/pull/91819
def __nn_Module_init_updated__(function):
    @functools.wraps(function)
    def new_function(self, *args, **kwargs):
        function(self, *args, **kwargs)
        super(nn.Module, self).__init__(*args, **kwargs)

    return new_function


nn.Module.__init__ = __nn_Module_init_updated__(nn.Module.__init__)


class Layer(nn.Module):
    _inputs: Union[None, ArgsKwargs]
    _outputs: Union[None, Tensor, Sequence[Tensor]]
    _backward_module: Optional[BackwardModule]
    _use_autograd_graph: bool
    graphs: Optional[ModelGraph]

    def __init__(self):
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module = None
        self._use_autograd_graph = False
        self.graphs = None

    def __call__(self, *inputs, **kwargs):
        self._forward_wrapper(self.forward)
        outputs = super(Layer, self).__call__(*inputs, **kwargs)
        if self.training:
            self._inputs = ArgsKwargs(args=inputs, kwargs=kwargs)
            self._outputs = outputs

        return outputs

    @property
    def use_autograd_graph(self):
        if self.graphs is not None:
            return self.graphs.use_autograd_graph
        return self._use_autograd_graph

    @use_autograd_graph.setter
    def use_autograd_graph(self, use_autograd_graph: bool):
        self._use_autograd_graph = use_autograd_graph
        if self.graphs is not None:
            self.graphs.use_autograd_graph = use_autograd_graph

    @property
    def inputs(self):
        return ArgsKwargs.from_args_kwargs_object(self._inputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def backward_function(self) -> Optional[BackwardModule]:
        if self._backward_module is not None:
            return self._backward_module

        if isinstance(self, BackwardModule):
            return self

        return None

    @backward_function.setter
    def backward_function(self, function):
        self.set_backward_function(function)

    def set_backward_function(self, backward_class: Union[BackwardModule, Type[BackwardModule], Callable]) -> Layer:
        if backward_class == self:
            return self

        if issubclass(backward_class, BackwardModule):
            self._backward_module = backward_class(self)
        elif isinstance(backward_class, BackwardModule):
            backward_class.set_layer(self)
            self._backward_module = backward_class
        elif callable(backward_class):
            self._backward_module = BackwardFunction(backward_class, self)
        else:
            raise Exception(f"Backward Module is not set for '{self}'")

        return self

    def _forward_wrapper(self, function: Callable):
        if hasattr(function, "__wrapper__") and function.__wrapper__ == Layer._forward_wrapper:
            return function

        if not isinstance(self.backward_function, BackwardModule):
            return function

        if not self.backward_function.has_forward():
            self.backward_function.forward = self.forward

        @functools.wraps(function)
        def new_forward(*args, **kwargs):
            return self.backward_function.auto_apply(*args, to_apply=self.use_autograd_graph, **kwargs)

        new_forward.__wrapped__ = function
        new_forward.__wrapper__ = Layer._forward_wrapper
        setattr(self, "forward", new_forward)
        return new_forward

    def _call_impl_forward(self, *args, **kwargs):
        if isinstance(self.backward_function, BackwardModule) and self.backward_function.has_forward():
            forward_functions = self.backward_function.forward
        else:
            forward_functions = self.forward

        if hasattr(forward_functions, "__wrapped__"):
            forward_functions = forward_functions.__wrapped__

        return forward_functions(*args, **kwargs)

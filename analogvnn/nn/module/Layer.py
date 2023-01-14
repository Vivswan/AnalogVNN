from __future__ import annotations

import functools
from typing import Union, Type, Callable, Sequence, Optional, TYPE_CHECKING, Set, Iterator, Tuple

from torch import nn, Tensor

from analogvnn.backward.BackwardFunction import BackwardFunction
from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.graph.ArgsKwargs import ArgsKwargs, ArgsKwargsOutput
from analogvnn.utils.common_types import TENSORS

if TYPE_CHECKING:
    from analogvnn.graph.ModelGraph import ModelGraph

__all__ = ['Layer']


# https://github.com/pytorch/pytorch/pull/91819
def __nn_Module_init_updated__(function: Callable) -> Callable:
    """Wrapper for nn.Module.__init__ to support multiple parent classes at same time.

    Args:
        function (Callable): nn.Module.__init__ function

    Returns:
        Callable: Wrapped function
    """
    # noinspection PyUnusedLocal
    def _temp(*args, **kwargs) -> ...:
        pass

    @functools.wraps(function)
    def new_function(self, *args, **kwargs):
        super_init = None
        next_mro_index = self.__class__.__mro__.index(nn.Module) + 1
        next_mro_class = self.__class__.__mro__[next_mro_index]

        if next_mro_class is not object:
            super_init = next_mro_class.__init__
            next_mro_class.__init__ = _temp

        function(self, *args, **kwargs)

        if next_mro_class is not object:
            next_mro_class.__init__ = super_init
            super(nn.Module, self).__init__()

    return new_function


if not hasattr(nn.Module, 'call_super_init'):
    nn.Module.__init__ = __nn_Module_init_updated__(nn.Module.__init__)
    """nn.Module.__init__ is updated to support multiple parent classes at same time. """


class Layer(nn.Module):
    """Base class for analog neural network modules.

    Attributes:
        _inputs (Union[None, ArgsKwargs]): Inputs of the layer.
        _outputs (Union[None, Tensor, Sequence[Tensor]]): Outputs of the layer.
        _backward_module (Optional[BackwardModule]): Backward module of the layer.
        _use_autograd_graph (bool): If True, the autograd graph is used to calculate the gradients.
        graphs (Optional[ModelGraph]): Contains Forward and Backward Graphs of the layer.
        call_super_init (bool): If True, the super class __init__ of nn.Module is called
        https://github.com/pytorch/pytorch/pull/91819
    """

    _inputs: Union[None, ArgsKwargs]
    _outputs: Union[None, Tensor, Sequence[Tensor]]
    _backward_module: Optional[BackwardModule]
    _use_autograd_graph: bool
    graphs: Optional[ModelGraph]

    # https://github.com/pytorch/pytorch/pull/91819
    call_super_init: bool = True

    def __init__(self):
        """Initializes the layer."""
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module = None
        self._use_autograd_graph = False
        self.graphs = None

    def __call__(self, *inputs, **kwargs):
        """Calls the forward pass of neural network layer.

        Args:
            *inputs: Inputs of the forward pass.
            **kwargs: Keyword arguments of the forward pass.
        """
        self._forward_wrapper(self.forward)
        outputs = super(Layer, self).__call__(*inputs, **kwargs)
        if self.training:
            self._inputs = ArgsKwargs(args=inputs, kwargs=kwargs)
            self._outputs = outputs

        return outputs

    @property
    def use_autograd_graph(self) -> bool:
        """If True, the autograd graph is used to calculate the gradients.

        Returns:
            bool: use_autograd_graph.
        """
        if self.graphs is not None:
            return self.graphs.use_autograd_graph
        return self._use_autograd_graph

    @use_autograd_graph.setter
    def use_autograd_graph(self, use_autograd_graph: bool):
        """Sets the use_autograd_graph attribute.

        Args:
            use_autograd_graph (bool): use_autograd_graph.
        """
        self._use_autograd_graph = use_autograd_graph
        if self.graphs is not None:
            self.graphs.use_autograd_graph = use_autograd_graph

    @property
    def inputs(self) -> ArgsKwargsOutput:
        """Inputs of the layer.

        Returns:
            ArgsKwargsOutput: inputs.
        """
        return ArgsKwargs.from_args_kwargs_object(self._inputs)

    @property
    def outputs(self) -> Union[None, Tensor, Sequence[Tensor]]:
        """Outputs of the layer.

        Returns:
            Union[None, Tensor, Sequence[Tensor]]: outputs.
        """
        return self._outputs

    @property
    def backward_function(self) -> Union[None, Callable, BackwardModule]:
        """Backward module of the layer.

        Returns:
            Union[None, Callable, BackwardModule]: backward_function.
        """
        if self._backward_module is not None:
            return self._backward_module

        if isinstance(self, BackwardModule):
            return self

        return None

    @backward_function.setter
    def backward_function(self, function: Union[BackwardModule, Type[BackwardModule], Callable]):
        """Sets the backward_function attribute.

        Args:
            function (Union[BackwardModule, Type[BackwardModule], Callable]): backward_function.
        """
        self.set_backward_function(function)

    def set_backward_function(self, backward_class: Union[Callable, BackwardModule, Type[BackwardModule]]) -> Layer:
        """Sets the backward_function attribute.

        Args:
            backward_class (Union[Callable, BackwardModule, Type[BackwardModule]]): backward_function.

        Returns:
            Layer: self.

        Raises:
            TypeError: If backward_class is not a callable or BackwardModule.
        """
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
            raise TypeError(f'Backward Module is not set for "{self}"')

        return self

    def named_registered_modules(
            self,
            memo: Optional[Set[nn.Module]] = None,
            prefix: str = '',
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        """Returns an iterator over all registered modules in the network, yielding
        both the name of the module and the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """
        if memo is None:
            memo = set()

        if self.backward_function != self:
            memo.add(self.backward_function)

        for name, module in super(Layer, self).named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate
        ):
            if module is self:
                continue

            yield name, module

    def registered_modules(self) -> Iterator[nn.Module]:
        """Returns an iterator over registered modules under self.

        Yields:
            nn.Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """
        for _, module in self.named_registered_modules():
            yield module

    def _forward_wrapper(self, function: Callable) -> Callable:
        """Wrapper for the forward function.

        Args:
            function (Callable): Forward function.

        Returns:
            Callable: Wrapped function.
        """
        # noinspection PyUnresolvedReferences
        if hasattr(function, '__wrapper__') and function.__wrapper__ == Layer._forward_wrapper:
            return function

        if not isinstance(self.backward_function, BackwardModule):
            return function

        if not self.backward_function.has_forward():
            self.backward_function.forward = self.forward

        @functools.wraps(function)
        def new_forward(*args, **kwargs):
            return self.backward_function.auto_apply(
                *args,
                to_apply=self.use_autograd_graph,
                **kwargs
            )

        new_forward.__wrapped__ = function
        new_forward.__wrapper__ = Layer._forward_wrapper
        self.forward = new_forward
        return new_forward

    def _call_impl_forward(self, *args: Tensor, **kwargs: Tensor) -> TENSORS:
        """Calls the forward pass of the layer.

        Args:
            *args: Inputs of the forward pass.
            **kwargs: Keyword arguments of the forward pass.

        Returns:
            TENSORS: Outputs of the forward pass.
        """
        if isinstance(self.backward_function, BackwardModule) and self.backward_function.has_forward():
            forward_functions = self.backward_function.forward
        else:
            forward_functions = self.forward

        if hasattr(forward_functions, '__wrapped__'):
            forward_functions = forward_functions.__wrapped__

        return forward_functions(*args, **kwargs)

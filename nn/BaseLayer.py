from typing import Union, Type

from torch import nn, Tensor


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self._saved_tensor = {}
        self._backward_module: Union[None, BackwardFunction] = None

    def get_backward_module(self) -> Union[None, 'BackwardFunction']:
        return self._backward_module

    def set_backward_module(self, backward_class: Type['BackwardFunction']) -> 'BaseLayer':
        if not issubclass(backward_class, BackwardFunction):
            raise Exception(f"Backward Module is not set for '{self}'")
        self._backward_module = backward_class(self)
        return self

    def use(self, *args) -> 'BaseLayer':
        for i in args:
            if issubclass(i, BackwardFunction):
                self.set_backward_module(i)
        return self

    def save_tensor(self, name: str, tensor: Tensor):
        if isinstance(tensor, Tensor):
            clone = tensor.clone()
            clone.detach_()
            clone.requires_grad = False
            self._saved_tensor[name] = clone

    def get_tensor(self, name: str) -> Union[None, Tensor]:
        if name in self._saved_tensor:
            return self._saved_tensor[name]
        else:
            return None

    def has_tensor(self, name: str) -> bool:
        return name in self._saved_tensor

    def clear_tensors(self):
        self._saved_tensor = {}


class BackwardFunction:
    __constants__ = ['main_layer']

    def __init__(self, layer: BaseLayer):
        self._layer = layer
        self.reset_parameters()

    def get_tensor(self, name: str) -> Union[None, Tensor]:
        if hasattr(self._layer, name):
            tensor = getattr(self._layer, name)
            if tensor is None or isinstance(tensor, Tensor):
                return tensor
            else:
                raise TypeError(f'"{name}" is not a tensor')
        elif isinstance(self._layer, BaseLayer) and self._layer.has_tensor(name):
            return self._layer.get_tensor(name)
        else:
            raise Exception(f'"{name}" is not found')

    def reset_parameters(self):
        pass

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        raise NotImplementedError

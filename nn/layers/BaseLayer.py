from typing import Union, Type

from torch import nn, Tensor


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self._saved_tensor = {}
        self._backward_module: Union[None, BackwardFunction] = None
        self._parent_module_attr = lambda x: None

    def set_backward_module(self, backward_class: Type['BackwardFunction']) -> 'BaseLayer':
        if not issubclass(backward_class, BackwardFunction):
            raise Exception(f"Backward Module is not set for '{self}'")
        self._backward_module = backward_class(self)
        return self

    def get_backward_module(self) -> Union[None, 'BackwardFunction']:
        return self._backward_module

    def use(self, *args) -> 'BaseLayer':
        for i in args:
            if issubclass(i, BackwardFunction):
                self.set_backward_module(i)
        return self

    def save_tensor(self, name: str, tensor: Tensor, attached=False):
        if isinstance(tensor, Tensor) and not attached:
            tensor = tensor.clone()
            tensor.detach_()
            tensor.requires_grad = False
        self._saved_tensor[name] = tensor
        return tensor

    def save_x(self, x: Tensor, attached=False):
        return self.save_tensor("input", x, attached=attached)

    def save_y(self, y: Tensor, attached=False):
        return self.save_tensor("output", y, attached=attached)

    def save_xy(self, x: Tensor, y: Tensor, attached=False):
        return self.save_x(x, attached=attached), self.save_y(y, attached=attached)

    def get_tensor(self, name: str) -> Union[None, Tensor]:
        if self.has_tensor(name):
            return self._saved_tensor[name]
        else:
            return None

    @property
    def x(self):
        return self.get_tensor("input")

    @property
    def y(self):
        return self.get_tensor("output")

    def has_tensor(self, name: str) -> bool:
        return name in self._saved_tensor

    def clear_tensors(self):
        self._saved_tensor = {}


class BackwardFunction:
    def __init__(self, layer: BaseLayer):
        if not isinstance(layer, BaseLayer):
            raise Exception(f'layer not instance of BaseLayer class')

        self._layer = layer
        self.reset_parameters()

    def get_parameter(self, name: str) -> Union[None, Tensor]:
        if self._layer.has_tensor(name):
            return self._layer.get_tensor(name)

        if hasattr(self._layer, name):
            return getattr(self._layer, name)

        raise Exception(f'"{name}" is not found')

    def reset_parameters(self):
        pass

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        raise NotImplementedError

    @property
    def x(self):
        return self.get_parameter("input")

    @property
    def y(self):
        return self.get_parameter("output")

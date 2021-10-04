from typing import Union

from torch import Tensor

from nn.BaseLayer import BaseLayer


class BackwardFunction:
    __constants__ = ['main_layer']

    def __init__(self, layer: 'BaseLayer'):
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


class BackwardUsingForward(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        mode = self._layer.training

        self._layer.training = True
        result = self._layer.forward(grad_output)
        self._layer.training = mode

        return result


class BackwardIdentity(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output

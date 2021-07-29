from typing import Union, Callable

from torch import nn, Tensor


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self._default_backward = None

    def set_default_backward(self, fn: Callable[[Union[None, Tensor]], Union[None, Tensor]]):
        self._default_backward = fn

    def get_default_backward(self):
        return self._default_backward
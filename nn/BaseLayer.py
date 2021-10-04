from typing import Union

from torch import nn, Tensor

from nn.backward_pass.BackwardFunction import BackwardUsingForward, BackwardIdentity


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self.backward_with_forward = BackwardUsingForward(self)
        self.backward_identity = BackwardIdentity(self)
        self._saved_tensor = {}

    def save_tensor(self, name: str, tensor: Tensor):
        if isinstance(tensor, Tensor):
            clone = tensor.clone()
            clone.detach_()
            clone.requires_grad = False
            self._saved_tensor[name] = clone
        else:
            raise TypeError(f'expected: Tensor, found: {tensor}')

    def get_tensor(self, name: str) -> Union[None, Tensor]:
        if name in self._saved_tensor:
            return self._saved_tensor[name]
        else:
            return None

    def has_tensor(self, name: str) -> bool:
        return name in self._saved_tensor

    def clear_tensors(self):
        self._saved_tensor = {}

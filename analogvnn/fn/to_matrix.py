from torch import Tensor

__all__ = ['to_matrix']


def to_matrix(tensor: Tensor):
    if len(tensor.size()) == 1:
        temp: Tensor = tensor.reshape(tuple([1] + list(tensor.size())))
        temp.requires_grad = tensor.requires_grad
        tensor = temp
    return tensor

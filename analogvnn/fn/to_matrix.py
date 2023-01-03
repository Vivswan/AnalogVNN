from torch import Tensor


def to_matrix(tensor: Tensor):
    if len(tensor.size()) == 1:
        temp: Tensor = tensor.reshape(tuple([1] + list(tensor.size())))
        temp.requires_grad = tensor.requires_grad
        tensor = temp
    return tensor

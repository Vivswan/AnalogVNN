from torch import Tensor


def to_matrix(tensor: Tensor):
    if len(tensor.size()) == 1:
        tensor = tensor.reshape(tuple([1] + list(tensor.size())))
    return tensor

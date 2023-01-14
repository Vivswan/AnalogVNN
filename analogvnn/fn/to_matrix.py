from torch import Tensor

__all__ = ['to_matrix']


def to_matrix(tensor: Tensor) -> Tensor:
    """`to_matrix` takes a tensor and returns a matrix with the same values as the tensor.

    Args:
        tensor (Tensor): Tensor

    Returns:
        Tensor: Tensor with the same values as the tensor, but with shape (1, -1).
    """

    if len(tensor.size()) == 1:
        return tensor.view(1, -1)
    return tensor

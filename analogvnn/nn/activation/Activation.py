import abc

from torch import Tensor, nn

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.nn.module.Layer import Layer

__all__ = ['Activation', 'InitImplement']


class InitImplement:
    """Implements the initialisation of parameters using the activation function."""

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """Initialisation of tensor using xavier uniform initialisation.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.xavier_uniform(tensor)

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """In-place initialisation of tensor using xavier uniform initialisation.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.xavier_uniform_(tensor)


class Activation(Layer, BackwardModule, InitImplement, abc.ABC):
    """This class is base class for all activation functions."""

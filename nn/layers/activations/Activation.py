import abc
from abc import ABC

from torch import Tensor, nn

from nn.graphs.BackwardFunction import BackwardFunction
from nn.modules.Layer import Layer


class InitImplement:
    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor)

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor)


class Activation(Layer, BackwardFunction, InitImplement, metaclass=abc.ABCMeta):
    pass

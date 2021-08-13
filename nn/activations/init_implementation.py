from torch import Tensor


class InitImplement:
    def initialise(self, tensor: Tensor) -> Tensor:
        raise NotImplemented

    def initialise_(self, tensor: Tensor) -> Tensor:
        raise NotImplemented

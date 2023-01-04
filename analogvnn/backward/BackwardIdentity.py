from typing import Sequence, Union

from torch import Tensor

from analogvnn.backward.BackwardModule import BackwardModule


class BackwardIdentity(BackwardModule):
    def backward(self, *grad_output) -> Union[None, Tensor, Sequence[Tensor]]:
        if len(grad_output) == 0:
            return None

        if len(grad_output) == 1:
            return grad_output[0]

        return grad_output

from typing import Optional

from torch import Tensor

from analogvnn.backward.BackwardModule import BackwardModule


class BackwardIdentity(BackwardModule):
    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return grad_output

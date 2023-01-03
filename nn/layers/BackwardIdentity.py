from typing import Optional

from torch import Tensor

from nn.modules.BackwardModule import BackwardModule


class BackwardIdentity(BackwardModule):
    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return grad_output

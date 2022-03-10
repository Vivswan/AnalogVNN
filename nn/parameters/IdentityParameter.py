import torch

from nn.parameters.BaseParameter import BaseParameter


class IdentityParameter(BaseParameter):
    def __repr__(self):
        return f'IdentityParameter: ' + super(IdentityParameter, self).__repr__()

    @torch.no_grad()
    def set_tensor(self, data):
        self.data = data
        return self

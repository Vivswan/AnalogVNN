import torch

from nn.parameters.Parameter import Parameter


class IdentityParameter(Parameter):
    def __repr__(self):
        return f'IdentityParameter: ' + super(IdentityParameter, self).__repr__()

    @torch.no_grad()
    def set_tensor(self, data):
        self.data = data
        return self

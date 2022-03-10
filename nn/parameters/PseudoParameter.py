import torch

from nn.parameters.BaseParameter import BaseParameter


class PseudoParameter(BaseParameter):
    def __init__(self, data=None, requires_grad=True, initialise_zero_pseudo=False):
        super(PseudoParameter, self).__init__(data, requires_grad)

        self.pseudo_tensor = None
        self.initialise_zero_pseudo = initialise_zero_pseudo
        self.initialise(data)

    def initialise(self, data):
        self.data = data

        if self.initialise_zero_pseudo:
            self.pseudo_tensor = torch.zeros_like(self.data, requires_grad=False)
        else:
            self.pseudo_tensor = torch.clone(self.data)
            self.pseudo_tensor.detach_()

        self.pseudo_tensor.parent = self
        self.pseudo_tensor.requires_grad_(False)
        return self

    def __repr__(self):
        return f'PseudoParameter(initialise_zero_pseudo:{self.initialise_zero_pseudo}):\n' \
               + super(PseudoParameter, self).__repr__()


if __name__ == '__main__':
    p = PseudoParameter(data=torch.eye(2))
    print(p)
    print(p.data)
    print(p.pseudo_tensor)
    print(p + torch.eye(2))
    p = p * 3
    print(p)

import torch

from nn.parameters.BaseParameter import BaseParameter


class BasePrecisionParameter(BaseParameter):
    def __init__(self, data=None, requires_grad=True, use_zero_pseudo_tensor=False):
        super(BasePrecisionParameter, self).__init__(data, requires_grad)

        if use_zero_pseudo_tensor:
            self.pseudo_tensor = torch.zeros_like(self.data, requires_grad=False)
        else:
            self.pseudo_tensor = torch.clone(self.data)
            self.pseudo_tensor.detach_()

        self.pseudo_tensor.parent = self
        self.pseudo_tensor.requires_grad_(False)

        self.use_zero_pseudo_tensor = use_zero_pseudo_tensor
        self.set_tensor(self.data)

    def __repr__(self):
        return f'BasePrecisionParameter(use_zero_pseudo_tensor:{self.use_zero_pseudo_tensor}): ' + super(BaseParameter,
                                                                                                         self).__repr__()

    @torch.no_grad()
    def set_tensor(self, data, *args, **kwargs):
        raise NotImplementedError

from torch import nn


class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, *args, **kwargs):
        return super(Parameter, cls).__new__(cls, data, requires_grad)

    # noinspection PyUnusedLocal
    def __init__(self, data=None, requires_grad=True, *args, **kwargs):
        super(Parameter, self).__init__()

    def __repr__(self, *args, **kwargs):
        return super(Parameter, self).__repr__(*args, **kwargs)

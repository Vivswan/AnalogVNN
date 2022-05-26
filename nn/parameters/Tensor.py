import torch


class Tensor(torch.Tensor):
    def __new__(cls, data=None, *args, **kwargs):
        return super(Tensor, cls).__new__(cls, data)

    # noinspection PyUnusedLocal
    def __init__(self, data=None, *args, **kwargs):
        super(Tensor, self).__init__()

from torch import nn

from nn.backward_pass_hook import BackwardPass


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self.backward = BackwardPass(self)

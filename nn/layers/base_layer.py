from torch import nn


class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()

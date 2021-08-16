import numpy as np
import torch.nn as nn

from nn.BaseModel import BaseModel
from nn.layers.Linear import Linear
from runs.r__._apporaches import BackPassTypes


class SingleLinearLayerModel(BaseModel):
    approaches = [BackPassTypes.default, BackPassTypes.BP]

    def __init__(self, in_features: tuple, out_features: int, approach, activation_class=None):
        super().__init__()
        self.approach = approach
        self.std = None
        self.activation_class = activation_class

        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = Linear(int(np.prod(in_features[1:])), out_features,
                              activation=None if activation_class is None else activation_class())
        self.log_softmax = nn.LogSoftmax(dim=1)

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True
        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT, self.linear1.backpropagation)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.log_softmax(x)
        return x

    def extra_repr(self) -> str:
        return f'approach={self.approach}'

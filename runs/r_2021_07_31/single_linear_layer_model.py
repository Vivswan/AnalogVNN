import numpy as np
import torch.nn as nn

from nn.activations.relu import ReLU
from nn.layers.linear import Linear
from nn.model_base import BaseModel
from runs.r_2021_07_31._apporaches import BackPassTypes


class SingleLinearLayerModel(BaseModel):
    approaches = [BackPassTypes.default, BackPassTypes.BP]

    def __init__(self, in_features: tuple, out_features: int, approach):
        super().__init__()
        self.approach = approach

        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = Linear(int(np.prod(in_features[1:])), out_features)
        self.relu1 = ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True
        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT, self.relu1, self.linear1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.log_softmax(x)
        return x

    def extra_repr(self) -> str:
        return f'approach={self.approach}'

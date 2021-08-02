import numpy as np
import torch.nn as nn

from nn.activations.relu import ReLU
from nn.layers.linear import Linear
from nn.model_base import BaseModel
from runs.r_2021_07_31._apporaches import BackPassTypes


class TripleLinearLayerModel(BaseModel):
    approaches = [BackPassTypes.default, BackPassTypes.BP, BackPassTypes.FA, BackPassTypes.RFA, BackPassTypes.DFA,
                  BackPassTypes.RDFA]

    def __init__(self, in_features: tuple, out_features: int, approach, std=None):
        super().__init__()
        self.approach = approach
        self.std = std

        in_size = int(np.prod(in_features[1:]))

        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = Linear(in_size, int(in_size / 2))
        self.relu1 = ReLU()
        self.linear2 = Linear(int(in_size / 2), int(in_size / 4))
        self.relu2 = ReLU()
        self.linear3 = Linear(int(in_size / 4), out_features)
        self.relu3 = ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.relu3, self.linear3.backpropagation,
                                       self.relu2, self.linear2.backpropagation,
                                       self.relu1, self.linear1.backpropagation)

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(self.backward.OUTPUT,
                                       self.relu3, self.linear3.feedforward_alignment,
                                       self.relu2, self.linear2.feedforward_alignment,
                                       self.relu1, self.linear1.feedforward_alignment)

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(self.backward.OUTPUT, self.relu3, self.linear3.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.relu2, self.linear2.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.relu1, self.linear1.direct_feedforward_alignment)

        if approach == BackPassTypes.RFA or approach == BackPassTypes.RDFA:
            self.linear1.feedforward_alignment.is_fixed = False
            self.linear2.feedforward_alignment.is_fixed = False
            self.linear3.feedforward_alignment.is_fixed = False
            self.linear1.direct_feedforward_alignment.is_fixed = False
            self.linear2.direct_feedforward_alignment.is_fixed = False
            self.linear3.direct_feedforward_alignment.is_fixed = False

        if std is not None:
            self.linear1.feedforward_alignment.std = std
            self.linear2.feedforward_alignment.std = std
            self.linear3.feedforward_alignment.std = std
            self.linear1.direct_feedforward_alignment.std = std
            self.linear2.direct_feedforward_alignment.std = std
            self.linear3.direct_feedforward_alignment.std = std

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.log_softmax(x)
        return x

    def extra_repr(self) -> str:
        return f'approach={self.approach}, std={self.std}'

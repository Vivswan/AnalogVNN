import numpy as np
import torch.nn as nn

from nn.BaseModel import BaseModel
from nn.layers.Linear import Linear
from runs.r__._apporaches import BackPassTypes


class TripleLinearLayerModel(BaseModel):
    approaches = [BackPassTypes.default, BackPassTypes.BP, BackPassTypes.FA, BackPassTypes.RFA, BackPassTypes.DFA,
                  BackPassTypes.RDFA]

    def __init__(self, in_features: tuple, out_features: int, approach, std=None, activation_class=None):
        super().__init__()
        self.approach = approach
        self.std = std
        self.activation_class = activation_class

        in_size = int(np.prod(in_features[1:]))

        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = Linear(in_size, int(in_size / 2),
                              activation=None if activation_class is None else activation_class())
        self.linear2 = Linear(int(in_size / 2), int(in_size / 4),
                              activation=None if activation_class is None else activation_class())
        self.linear3 = Linear(int(in_size / 4), out_features,
                              activation=None if activation_class is None else activation_class())
        self.log_softmax = nn.LogSoftmax(dim=1)

        if approach == BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == BackPassTypes.BP:
            self.backward.add_relation(self.backward.OUTPUT, self.linear3.backpropagation, self.linear2.backpropagation,
                                       self.linear1.backpropagation)

        if approach == BackPassTypes.FA or approach == BackPassTypes.RFA:
            self.backward.add_relation(self.backward.OUTPUT, self.linear3.feedforward_alignment,
                                       self.linear2.feedforward_alignment, self.linear1.feedforward_alignment)

        if approach == BackPassTypes.DFA or approach == BackPassTypes.RDFA:
            self.backward.add_relation(self.backward.OUTPUT, self.linear3.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.linear2.direct_feedforward_alignment)
            self.backward.add_relation(self.backward.OUTPUT, self.linear1.direct_feedforward_alignment)

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
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.log_softmax(x)
        return x

    def extra_repr(self) -> str:
        return f'approach={self.approach}, std={self.std}'

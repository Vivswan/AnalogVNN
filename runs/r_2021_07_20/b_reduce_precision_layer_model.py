import numpy as np
import torch.nn as nn

from nn.BaseModel import BaseModel
from nn.layers.reduce_precision_layer import ReducePrecision


class ReducePrecisionLayerModel(BaseModel):
    def __init__(
            self,
            in_features: tuple,
            out_features: int,
            precision: int,
    ):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Flatten(start_dim=1),
            ReducePrecision(precision=precision),
            nn.Linear(int(np.prod(in_features[1:])), out_features),
            nn.ReLU(inplace=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.nn(x)

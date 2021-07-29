import numpy as np
import torch
import torch.nn as nn

from nn.model_base import BaseModel


class SingleLinearLayerModel(BaseModel):
    def __init__(
            self,
            in_features: tuple,
            out_features: int,
            device: torch.device,
    ):
        super().__init__(device)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_nn = nn.Sequential(
            nn.Linear(int(np.prod(in_features[1:])), out_features),
            nn.ReLU(inplace=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_nn(x)
        return x

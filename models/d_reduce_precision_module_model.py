import numpy as np
import torch
import torch.nn as nn

from nn.layers.reduce_precision_module import ReducePrecisionModule
from nn.model_base import ModelBase
from nn.summary import summary
from utils.is_using_cuda import is_using_cuda


class ReducePrecisionModuleModel(ModelBase):
    def __init__(
            self,
            in_features: tuple,
            out_features: int,
            device: torch.device,
            log_dir: str,
    ):
        super().__init__(in_features, log_dir, device)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_nn = nn.Sequential(
            ReducePrecisionModule(nn.Linear(int(np.prod(in_features[1:])), out_features)),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_nn(x)
        return x


if __name__ == '__main__':
    device, is_cuda = is_using_cuda()
    model = ReducePrecisionModuleModel(in_features=(1000, 24, 24), out_features=10, device=device,
                                       log_dir="D:/_data/tensorboard")
    print(summary(model))
    print(str(model))

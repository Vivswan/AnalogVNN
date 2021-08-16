from typing import Union, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from nn.BackwardPass import BackwardPass
from nn.BaseLayer import BaseLayer
from nn.TensorboardModelLog import TensorboardModelLog
from nn.test import test
from nn.train import train
from nn.utils.is_using_cuda import get_device

_grad_t = Union[Tuple[Tensor, ...], Tensor]


class BaseModel(nn.Module):
    __constants__ = ['in_features', 'device']

    device: torch.device
    tensorboard: Union[None, TensorboardModelLog]

    def __init__(self):
        super(BaseModel, self).__init__()

        self._compiled = False
        self.tensorboard = None
        self._output_hook = None
        self.backward = BackwardPass()
        self.optimizer = None
        self.loss = None
        self.accuracy = None
        self.device = get_device()

    def compile(self, device=get_device(), layer_data=True):
        self.backward.compile()
        self.to(device=device)
        self.device = device

        self._compiled = True
        if self.tensorboard is not None:
            self.tensorboard.on_compile(layer_data=layer_data)
        return self

    def output(self, x):
        result = self(x)
        self.backward.set_output(result)
        return result

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int = None):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        train_loss, train_accuracy = train(self, self.device, train_loader, self.optimizer, self.loss, epoch)
        test_loss, test_accuracy = test(self, self.device, test_loader, self.loss)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(train_loader)
            self.tensorboard.register_training(epoch, train_loss, train_accuracy)
            self.tensorboard.register_testing(epoch, test_loss, test_accuracy)

        return train_loss, train_accuracy, test_loss, test_accuracy

    def create_tensorboard(self, log_dir: str):
        self.tensorboard = TensorboardModelLog(self, log_dir=log_dir)
        self.subscribe_tensorboard(self.tensorboard)

    def subscribe_tensorboard(self, tensorboard: TensorboardModelLog):
        self.tensorboard = tensorboard
        if self._compiled is True:
            self.tensorboard.on_compile()

    def apply_to_parameters(self: nn.Module, layer: BaseLayer, requires_grad=True):
        with torch.no_grad():
            layer.train()
            for p in self.parameters():
                if requires_grad and not p.requires_grad:
                    continue
                p.data = layer.forward(p.data)

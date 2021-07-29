from typing import Union, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from nn.TensorboardModelLog import TensorboardModelLog
from nn.backward_pass_hook import BackwardPass
from nn.test import test
from nn.train import train
from nn.utils.is_using_cuda import get_device

_grad_t = Union[Tuple[Tensor, ...], Tensor]

class BaseModel(nn.Module):
    __constants__ = ['in_features', 'device']

    in_features: tuple
    device: torch.device

    tensorboard: Union[None, TensorboardModelLog]

    def __init__(self, in_features: tuple, device: torch.device = get_device()):
        super(BaseModel, self).__init__()

        self._compiled = False
        self.in_features = in_features
        self.device = device
        self.tensorboard = None
        self._output_hook = None
        self.backward = BackwardPass()

    def compile(self):
        self.backward.compile()

        self._compiled = True
        if self.tensorboard is not None:
            self.tensorboard.on_compile()
        return self

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int = None):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        train_loss, train_accuracy = train(self, self.device, train_loader, self.optimizer, self.loss_fn, epoch)
        test_loss, test_accuracy = test(self, self.device, test_loader, self.loss_fn)

        if self.tensorboard is not None:
            self.tensorboard.register_training(epoch, train_loss, train_accuracy)
            self.tensorboard.register_testing(epoch, test_loss, test_accuracy )

        return train_loss, train_accuracy, test_loss, test_accuracy


    def subscribe_tensorboard(self, tensorboard: TensorboardModelLog):
        self.tensorboard = tensorboard
        if self._compiled is True:
            self.tensorboard.on_compile()

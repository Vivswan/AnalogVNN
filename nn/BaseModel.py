from typing import Union, Type

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from nn.BaseLayer import BaseLayer
from nn.TensorboardModelLog import TensorboardModelLog
from nn.backward_pass.BaseBackwardPass import BaseBackwardPass
from nn.backward_pass.GraphBackwardPass import GraphBackwardPass
from nn.test import test
from nn.train import train
from nn.utils.is_using_cuda import get_device


class BaseModel(nn.Module):
    __constants__ = ['in_features', 'device']

    device: torch.device
    tensorboard: Union[None, TensorboardModelLog]

    def __init__(self, tensorboard_log_dir=None, backward_class: Type[BaseBackwardPass] = GraphBackwardPass):
        super(BaseModel, self).__init__()

        self._compiled = False
        self._output_hook = None

        self.tensorboard = None
        if tensorboard_log_dir is not None:
            self.create_tensorboard(tensorboard_log_dir)

        if not issubclass(backward_class, BaseBackwardPass):
            raise Exception(f"backward_class must be subclass of '{BaseBackwardPass.__name__}'")

        self.backward = backward_class()
        self.optimizer = None
        self.loss_fn = None
        self.accuracy_fn = None
        self.device = get_device()

    def compile(self, device=get_device(), layer_data=True):
        self.backward.compile()
        self.to(device=device)
        self.device = device

        self._compiled = True
        if self.tensorboard is not None:
            self.tensorboard.on_compile(layer_data=layer_data)
        return self

    def output(self, x) -> Tensor:
        result = self(x)
        if self.training:
            self.backward.set_output(result)
        return result

    def loss(self, x, y):
        if self.loss_fn is None:
            raise Exception("loss_fn is not set")

        loss_result = self.loss_fn(x, y)
        if self.training:
            self.backward.set_loss(loss_result)

        accuracy_result = None
        if self.accuracy_fn is not None:
            accuracy_result = self.accuracy_fn(x, y)

        return loss_result, accuracy_result

    def apply_to_parameters(self: nn.Module, layer: BaseLayer, requires_grad=True):
        with torch.no_grad():
            layer.train()
            for p in self.parameters():
                if requires_grad and not p.requires_grad:
                    continue
                p.data = layer.forward(p.data)

    def train_on(self, train_loader: DataLoader, epoch: int = None):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        train_loss, train_accuracy = train(self, train_loader, epoch)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(train_loader)
            self.tensorboard.register_training(epoch, train_loss, train_accuracy)

        return train_loss, train_accuracy

    def test_on(self, test_loader: DataLoader, epoch: int = None):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        test_loss, test_accuracy = test(self, test_loader)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(test_loader)
            self.tensorboard.register_testing(epoch, test_loss, test_accuracy)

        return test_loss, test_accuracy

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int = None):
        train_loss, train_accuracy = self.train_on(train_loader=train_loader, epoch=epoch)
        test_loss, test_accuracy = self.test_on(test_loader=test_loader, epoch=epoch)
        return train_loss, train_accuracy, test_loss, test_accuracy

    def create_tensorboard(self, log_dir: str):
        self.tensorboard = TensorboardModelLog(self, log_dir=log_dir)
        self.subscribe_tensorboard(self.tensorboard)

    def subscribe_tensorboard(self, tensorboard: TensorboardModelLog):
        self.tensorboard = tensorboard
        if self._compiled is True:
            self.tensorboard.on_compile()

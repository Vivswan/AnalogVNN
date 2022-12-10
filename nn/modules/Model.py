from typing import Union, Callable

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from nn.graphs.ModelGraph import ModelGraph
from nn.fn.test import test
from nn.fn.train import train
from nn.modules.Layer import Layer
from nn.utils.TensorboardModelLog import TensorboardModelLog
from nn.utils.is_cpu_cuda import is_cpu_cuda


class Model(Layer):
    __constants__ = ['in_features', 'device']

    device: torch.device

    # tensorboard: Union[None, TensorboardModelLog]

    def __init__(self, tensorboard_log_dir=None, device=is_cpu_cuda.get_device()):
        super(Model, self).__init__()

        self._compiled = False
        self._output_hook = None

        self.tensorboard = None
        if tensorboard_log_dir is not None:
            self.create_tensorboard(tensorboard_log_dir)

        self.graphs = ModelGraph()
        self.forward_graph = self.graphs.forward_graph
        self.backward_graph = self.graphs.backward_graph

        self.optimizer = None
        self.loss_function = None
        self.accuracy_function = None
        self.device = device

    def __call__(self, *args, **kwargs):
        result = super(Model, self).__call__(*args, **kwargs)
        if self.training:
            self.backward_graph.set_inputs(*args)
            result = self.backward_graph.set_output(result)
        return result

    def compile(self, device=None, layer_data=True):
        if device is not None:
            self.device = device

        self.graphs.compile()
        self.to(device=self.device)

        for module in self.children():
            if isinstance(module, Layer):
                module._parent_module_attr = lambda name: getattr(self, name) if hasattr(self, name) else None

        self._compiled = True
        if self.tensorboard is not None:
            self.tensorboard.on_compile(layer_data=layer_data)
        return self

    @property
    def use_autograd_graph(self):
        return self.backward_graph.use_autograd_graph

    @use_autograd_graph.setter
    def use_autograd_graph(self, use_autograd_graph: bool):
        self.backward_graph.use_autograd_graph = use_autograd_graph

    def backward(self, *inputs):
        return self.backward_graph(inputs)

    def output(self, x: Tensor) -> Tensor:
        return self(x)

    def loss(self, output, target):
        if self.loss_function is None:
            raise Exception("loss_function is not set")

        loss_result = self.loss_function(output, target)
        if self.training:
            self.backward_graph.set_loss(loss_result)

        accuracy_result = None
        if self.accuracy_function is not None:
            accuracy_result = self.accuracy_function(output, target)

        return loss_result, accuracy_result

    def apply_to_parameters(self: nn.Module, layer: Union[Layer, Callable], requires_grad=True):
        with torch.no_grad():
            for p in self.parameters():
                if requires_grad and not p.requires_grad:
                    continue
                p.data = layer(p.data)

    def train_on(self, train_loader: DataLoader, epoch: int = None, *args, **kwargs):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        train_loss, train_accuracy = train(self, train_loader, epoch, *args, **kwargs)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(train_loader)
            self.tensorboard.register_training(epoch, train_loss, train_accuracy)

        return train_loss, train_accuracy

    def test_on(self, test_loader: DataLoader, epoch: int = None, *args, **kwargs):
        if self._compiled is False:
            raise Exception("model is not complied yet")

        test_loss, test_accuracy = test(self, test_loader, *args, **kwargs)

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

    def forward(self, x: Tensor):
        raise NotImplementedError

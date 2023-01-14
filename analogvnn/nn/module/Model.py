from __future__ import annotations

import typing
from typing import Callable, Optional, Tuple, Set, Iterator

import torch
from torch import optim, Tensor, nn
from torch.utils.data import DataLoader

from analogvnn.fn.test import test
from analogvnn.fn.train import train
from analogvnn.graph.BackwardGraph import BackwardGraph
from analogvnn.graph.ForwardGraph import ForwardGraph
from analogvnn.graph.ModelGraph import ModelGraph
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.common_types import TENSORS
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda

if typing.TYPE_CHECKING:
    from analogvnn.utils.TensorboardModelLog import TensorboardModelLog

__all__ = ['Model']


class Model(Layer):
    """Base class for analog neural network models.

    Attributes:
        _compiled (bool): True if the model is compiled.
        tensorboard (TensorboardModelLog): The tensorboard logger of the model.
        graphs (ModelGraph): The graph of the model.
        forward_graph (ForwardGraph): The forward graph of the model.
        backward_graph (BackwardGraph): The backward graph of the model.
        optimizer (optim.Optimizer): The optimizer of the model.
        loss_function (Callable): The loss function of the model.
        accuracy_function (Callable): The accuracy function of the model.
        device (torch.device): The device of the model.
    """

    __constants__ = ['device']

    _compiled: bool

    tensorboard: Optional[TensorboardModelLog]

    graphs: ModelGraph
    forward_graph: ForwardGraph
    backward_graph: BackwardGraph

    optimizer: Optional[optim.Optimizer]
    loss_function: Optional[Callable]
    accuracy_function: Optional[Callable]
    device: torch.device

    def __init__(self, tensorboard_log_dir=None, device=is_cpu_cuda.device):
        """Create a new model.

        Args:
            tensorboard_log_dir (str): The log directory of the tensorboard logger.
            device (torch.device): The device to run the model on.
        """

        super(Model, self).__init__()

        self._compiled = False

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
        """Call the model.

        Args:
            *args: The arguments of the model.
            **kwargs: The keyword arguments of the model.

        Returns:
            TENSORS: The output of the model.

        Raises:
            RuntimeError: if the model is not compiled.
        """

        if not self._compiled:
            raise RuntimeError('Model is not compiled yet.')

        return super(Model, self).__call__(*args, **kwargs)

    @property
    def use_autograd_graph(self):
        """Is the autograd graph used for the model.

        Returns:
            bool: If True, the autograd graph is used to calculate the gradients.
        """

        return self.graphs.use_autograd_graph

    @use_autograd_graph.setter
    def use_autograd_graph(self, use_autograd_graph: bool):
        """Set if the autograd graph is used for the model.

        Args:
            use_autograd_graph (bool): If True, the autograd graph is used to calculate the gradients.
        """

        self.graphs.use_autograd_graph = use_autograd_graph

    def named_registered_modules(
            self,
            memo: Optional[Set[nn.Module]] = None,
            prefix: str = '',
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        """Returns an iterator over registered modules under self.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, nn.Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()

        memo.add(self.optimizer)
        memo.add(self.loss_function)
        memo.add(self.accuracy_function)
        return super(Model, self).named_registered_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)

    def compile(self, device: Optional[torch.device] = None, layer_data: bool = True):
        """Compile the model.

        Args:
            device (torch.device): The device to run the model on.
            layer_data (bool): If True, the layer data is logged.

        Returns:
            Model: The compiled model.
        """

        if device is not None:
            self.device = device

        self.graphs.compile()
        for i in self.modules():
            if isinstance(i, Layer) and i != self:
                i.graphs = self.graphs

        self.to(device=self.device)

        self._compiled = True
        if self.tensorboard is not None:
            self.tensorboard.on_compile(layer_data=layer_data)
        return self

    def forward(self, *inputs: Tensor) -> TENSORS:
        """Forward pass of the model.

        Args:
            *inputs (Tensor): The inputs of the model.

        Returns:
            TENSORS: The output of the model.
        """

        return self.graphs.forward_graph(inputs, self.training)

    @torch.no_grad()
    def backward(self, *inputs: Tensor) -> TENSORS:
        """Backward pass of the model.

        Args:
            *inputs (Tensor): The inputs of the model.

        Returns:
            TENSORS: The output of the model.
        """

        return self.graphs.backward_graph(inputs)

    def loss(self, output: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate the loss of the model.

        Args:
            output (Tensor): The output of the model.
            target (Tensor): The target of the model.

        Returns:
            Tuple[Tensor, Tensor]: The loss and the accuracy of the model.

        Raises:
            ValueError: if loss_function is None.
        """

        if self.loss_function is None:
            raise ValueError('loss_function is None')

        loss_result = self.loss_function(output, target)
        if self.training:
            self.graphs.set_loss(loss_result)

        accuracy_result = None
        if self.accuracy_function is not None:
            accuracy_result = self.accuracy_function(output, target)

        return loss_result, accuracy_result

    def train_on(self, train_loader: DataLoader, epoch: int = None, *args, **kwargs) -> Tuple[float, float]:
        """Train the model on the train_loader.

        Args:
            train_loader (DataLoader): The train loader of the model.
            epoch (int): The epoch of the model.
            *args: The arguments of the train function.
            **kwargs: The keyword arguments of the train function.

        Returns:
            Tuple[float, float]: The loss and the accuracy of the model.

        Raises:
            RuntimeError: if model is not compiled.
        """

        if self._compiled is False:
            raise RuntimeError('Model is not compiled')

        train_loss, train_accuracy = train(self, train_loader, epoch, *args, **kwargs)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(train_loader)
            self.tensorboard.register_training(epoch, train_loss, train_accuracy)

        return train_loss, train_accuracy

    def test_on(self, test_loader: DataLoader, epoch: int = None, *args, **kwargs) -> Tuple[float, float]:
        """Test the model on the test_loader.

        Args:
            test_loader (DataLoader): The test loader of the model.
            epoch (int): The epoch of the model.
            *args: The arguments of the test function.
            **kwargs: The keyword arguments of the test function.

        Returns:
            Tuple[float, float]: The loss and the accuracy of the model.

        Raises:
            RuntimeError: if model is not compiled.
        """

        if self._compiled is False:
            raise RuntimeError('Model is not compiled')

        test_loss, test_accuracy = test(self, test_loader, *args, **kwargs)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(test_loader)
            self.tensorboard.register_testing(epoch, test_loss, test_accuracy)

        return test_loss, test_accuracy

    def fit(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epoch: int = None
    ) -> Tuple[float, float, float, float]:
        """Fit the model on the train_loader and test the model on the test_loader.

        Args:
            train_loader (DataLoader): The train loader of the model.
            test_loader (DataLoader): The test loader of the model.
            epoch (int): The epoch of the model.

        Returns:
            Tuple[float, float, float, float]: The train loss, the train accuracy, the test loss
            and the test accuracy of the model.
        """

        train_loss, train_accuracy = self.train_on(train_loader=train_loader, epoch=epoch)
        test_loss, test_accuracy = self.test_on(test_loader=test_loader, epoch=epoch)
        return train_loss, train_accuracy, test_loss, test_accuracy

    def create_tensorboard(self, log_dir: str) -> TensorboardModelLog:
        """Create a tensorboard.

        Args:
            log_dir (str): The log directory of the tensorboard.

        Raises:
            ImportError: if tensorboard (https://www.tensorflow.org/) is not installed.
        """

        try:
            from analogvnn.utils.TensorboardModelLog import TensorboardModelLog
        except ImportError as e:
            raise ImportError('requires tensorboard https://www.tensorflow.org/') from e

        self.tensorboard = TensorboardModelLog(self, log_dir=log_dir)
        self.subscribe_tensorboard(self.tensorboard)
        return self.tensorboard

    def subscribe_tensorboard(self, tensorboard: TensorboardModelLog):
        """Subscribe the model to the tensorboard.

        Args:
            tensorboard (TensorboardModelLog): The tensorboard of the model.

        Returns:
            Model: self.
        """

        self.tensorboard = tensorboard
        if self._compiled is True:
            self.tensorboard.on_compile()

        return self

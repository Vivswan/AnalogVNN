from __future__ import annotations

import os
import re
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analogvnn.utils.summary import summary

__all__ = ['TensorboardModelLog']


class TensorboardModelLog:
    """Tensorboard model log.

    Attributes:
        model (nn.Module): the model to log.
        tensorboard (SummaryWriter): the tensorboard.
        layer_data (bool): whether to log the layer data.
    """
    model: nn.Module
    tensorboard: Optional[SummaryWriter]
    layer_data: bool

    def __init__(self, model: nn.Module, log_dir: str):
        """Log the model to Tensorboard.

        Args:
            model (nn.Module): the model to log.
            log_dir (str): the directory to log to.
        """
        self.model = model
        self.tensorboard = None
        self.layer_data = True

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.set_log_dir(log_dir)
        if hasattr(model, "subscribe_tensorboard"):
            model.subscribe_tensorboard(tensorboard=self)

    def set_log_dir(self, log_dir: str) -> TensorboardModelLog:
        """Set the log directory.

        Args:
            log_dir (str): the log directory.

        Returns:
            TensorboardModelLog: self.
        """
        if os.path.isdir(log_dir):
            self.tensorboard = SummaryWriter(log_dir=log_dir)
        else:
            raise Exception(f'"{log_dir}" is not a directory.')
        return self

    def _add_layer_data(self, epoch: int = None):
        """Add the layer data to the tensorboard.

        Args:
            epoch (int): the epoch to add the data for.
        """
        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList) or (module == self):
                continue

            idx += 1
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                self.tensorboard.add_histogram(f"{idx}-{module}.bias", module.bias, epoch)
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                self.tensorboard.add_histogram(f"{idx}-{module}.weight", module.weight, epoch)

    def on_compile(self, layer_data: bool = True):
        """Called when the model is compiled.

        Args:
            layer_data (bool): whether to log the layer data.
        """
        if self.layer_data:
            self.layer_data = layer_data

        if self.layer_data:
            self._add_layer_data(epoch=-1)
        return self

    def add_graph(self, train_loader: DataLoader, model: Optional[nn.Module] = None) -> TensorboardModelLog:
        """Add the model graph to the tensorboard.

        Args:
            train_loader (DataLoader): the train loader.
            model (nn.Module): the model to log.

        Returns:
            TensorboardModelLog: self.
        """
        if model is None:
            model = self.model

        if not getattr(self.tensorboard, f"_added_graph_{id(model)}", False):
            # print(f"_added_graph_{id(model)}", model.__class__.__name__)
            for batch_idx, (data, target) in enumerate(train_loader):
                input_size = tuple(list(data.shape)[1:])
                batch_size = data.shape[1]
                self.tensorboard.add_text(
                    f"str ({model.__class__.__name__})",
                    re.sub("\n", "\n    ", "    " + str(model))
                )
                self.tensorboard.add_text(
                    f"summary ({model.__class__.__name__})",
                    re.sub("\n", "\n    ", "    " + summary(model, input_size=input_size))
                )
                self.tensorboard.add_graph(model, torch.zeros(tuple([batch_size] + list(input_size))).to(model.device))
                break

            setattr(self.tensorboard, f"_added_graph_{id(model)}", True)
        return self

    def register_training(self, epoch: int, train_loss: float, train_accuracy: float) -> TensorboardModelLog:
        """Register the training data.

        Args:
            epoch (int): the epoch.
            train_loss (float): the training loss.
            train_accuracy (float): the training accuracy.

        Returns:
            TensorboardModelLog: self.
        """
        self.tensorboard.add_scalar('Loss/train', train_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/train", train_accuracy, epoch)
        if self.layer_data:
            self._add_layer_data(epoch=epoch)
        return self

    def register_testing(self, epoch: int, test_loss: float, test_accuracy: float) -> TensorboardModelLog:
        """Register the testing data.

        Args:
            epoch (int): the epoch.
            test_loss (float): the test loss.
            test_accuracy (float): the test accuracy.

        Returns:
            TensorboardModelLog: self.
        """
        self.tensorboard.add_scalar('Loss/test', test_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/test", test_accuracy, epoch)
        return self

    def close(self) -> TensorboardModelLog:
        """Close the tensorboard.

        Returns:
            TensorboardModelLog: self.
        """
        if self.tensorboard is not None:
            self.tensorboard.close()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

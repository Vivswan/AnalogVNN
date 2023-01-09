from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.module.Model import Model

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

    def __init__(self, model: Model, log_dir: str):
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

        Raises:
            ValueError: if the log directory is invalid.
        """

        # https://github.com/tensorflow/tensorboard/pull/6135
        from tensorboard.compat import tf
        if getattr(tf, "io", None) is None:
            import tensorboard.compat.tensorflow_stub as new_tf
            tf.__dict__.update(new_tf.__dict__)

        if os.path.isdir(log_dir):
            self.tensorboard = SummaryWriter(log_dir=log_dir)
        else:
            raise ValueError(f"Log directory {log_dir} does not exist.")
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

    def add_graph(
            self,
            train_loader: DataLoader,
            model: Optional[nn.Module] = None,
            input_size: Optional[Sequence[int]] = None,
    ) -> TensorboardModelLog:
        """Add the model graph to the tensorboard.

        Args:
            train_loader (DataLoader): the train loader.
            model (Optional[nn.Module]): the model to log.
            input_size (Optional[Sequence[int]]): the input size.

        Returns:
            TensorboardModelLog: self.
        """
        if model is None:
            model = self.model

        log_id = f"{TensorboardModelLog.add_graph.__name__}_{id(model)}"
        if getattr(self.tensorboard, log_id, False):
            return self

        if input_size is None:
            data_shape = next(iter(train_loader))[0].shape
            input_size = [1] + list(data_shape)[1:]

        if isinstance(model, Layer):
            model.use_autograd_graph = True

        self.tensorboard.add_graph(model, torch.zeros(input_size).to(model.device))
        setattr(self.tensorboard, log_id, True)

        if isinstance(model, Layer):
            model.use_autograd_graph = False

        return self

    def add_summary(
            self,
            train_loader: DataLoader,
            model: Optional[nn.Module] = None,
            input_size: Optional[Sequence[int]] = None,
    ) -> TensorboardModelLog:
        """Add the model summary to the tensorboard.

        Args:
            train_loader (DataLoader): the train loader.
            model (nn.Module): the model to log.
            input_size (Optional[Sequence[int]]): the input size.

        Returns:
            TensorboardModelLog: self.

        Raises:
            ImportError: if torchinfo (https://github.com/tyleryep/torchinfo) is not installed.
        """

        try:
            import torchinfo
        except ImportError as e:
            raise ImportError("requires torchinfo: https://github.com/tyleryep/torchinfo") from e

        if model is None:
            model = self.model

        log_id = f"{TensorboardModelLog.add_summary.__name__}_{id(model)}"
        if getattr(self.tensorboard, log_id, False):
            return self

        if input_size is None:
            data_shape = next(iter(train_loader))[0].shape
            input_size = tuple(list(data_shape)[1:])

        if isinstance(model, Layer):
            model.use_autograd_graph = True

        model_str = re.sub("\n", "\n    ", "    " + str(model))
        nn_model_summary = torchinfo.summary(
            model,
            input_size=input_size,
            verbose=torchinfo.Verbosity.QUIET,
            col_names=[e.value for e in torchinfo.ColumnSettings],
            depth=10,
        )
        nn_model_summary.formatting.verbose = torchinfo.Verbosity.VERBOSE
        nn_model_summary = re.sub("\n", "\n    ", f"    {nn_model_summary}")

        self.tensorboard.add_text(f"str ({model.__class__.__name__})", model_str)
        self.tensorboard.add_text(f"summary ({model.__class__.__name__})", nn_model_summary)
        setattr(self.tensorboard, log_id, True)

        if isinstance(model, Layer):
            model.use_autograd_graph = False
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

    # noinspection PyUnusedLocal
    def close(self, *args, **kwargs):
        """Close the tensorboard.
        """
        if self.tensorboard is not None:
            self.tensorboard.close()
            self.tensorboard = None

    __exit__ = close
    """Close the tensorboard."""
    __del__ = close
    """Close the tensorboard."""

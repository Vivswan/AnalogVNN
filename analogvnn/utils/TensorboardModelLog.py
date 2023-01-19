from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analogvnn.nn.module.Layer import Layer
from analogvnn.nn.module.Model import Model

__all__ = ['TensorboardModelLog']

from analogvnn.utils.get_model_summaries import get_model_summaries


class TensorboardModelLog:
    """Tensorboard model log.

    Attributes:
        model (nn.Module): the model to log.
        tensorboard (SummaryWriter): the tensorboard.
        layer_data (bool): whether to log the layer data.
        _log_record (Dict[str, bool]): the log record.
    """

    model: nn.Module
    tensorboard: Optional[SummaryWriter]
    layer_data: bool
    _log_record: Dict[str, bool]

    def __init__(self, model: Model, log_dir: str):
        """Log the model to Tensorboard.

        Args:
            model (nn.Module): the model to log.
            log_dir (str): the directory to log to.
        """

        super(TensorboardModelLog, self).__init__()
        self.model = model
        self.tensorboard = None
        self.layer_data = True
        self._log_record = {}

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.set_log_dir(log_dir)
        if hasattr(model, 'subscribe_tensorboard'):
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
        if getattr(tf, 'io', None) is None:
            import tensorboard.compat.tensorflow_stub as new_tf
            tf.__dict__.update(new_tf.__dict__)

        if os.path.isdir(log_dir):
            self.tensorboard = SummaryWriter(log_dir=log_dir)
        else:
            raise ValueError(f'Log directory {log_dir} does not exist.')
        return self

    def _add_layer_data(self, epoch: int = None):
        """Add the layer data to the tensorboard.

        Args:
            epoch (int): the epoch to add the data for.
        """

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            self.tensorboard.add_histogram(name, parameter.data, epoch)

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

        log_id = f'{self.tensorboard.log_dir}_{TensorboardModelLog.add_graph.__name__}_{id(model)}'
        if log_id in self._log_record:
            return self

        if input_size is None:
            data_shape = next(iter(train_loader))[0].shape
            input_size = [1] + list(data_shape)[1:]

        use_autograd_graph = False
        if isinstance(model, Layer):
            use_autograd_graph = model.use_autograd_graph
            model.use_autograd_graph = False

        graph_path = Path(self.tensorboard.log_dir).joinpath(f'graph_{model.__class__.__name__}_{id(model)}')
        with SummaryWriter(log_dir=str(graph_path)) as graph_writer:
            graph_writer.add_graph(model, torch.zeros(input_size).to(model.device))

        self._log_record[log_id] = True
        if isinstance(model, Layer):
            model.use_autograd_graph = use_autograd_graph

        return self

    def add_summary(
            self,
            model: Optional[nn.Module],
            input_size: Optional[Sequence[int]] = None,
            train_loader: Optional[DataLoader] = None,
            *args,
            **kwargs
    ) -> Tuple[str, str]:
        """Add the model summary to the tensorboard.

        Args:
            model (nn.Module): the model to log.
            input_size (Optional[Sequence[int]]): the input size.
            train_loader (Optional[DataLoader]): the train loader.
            *args: the arguments to torchinfo.summary.
            **kwargs: the keyword arguments to torchinfo.summary.

        Returns:
            Tuple[str, str]: the model __repr__ and the model summary.
        """

        if model is None:
            model = self.model

        log_id = f'{self.tensorboard.log_dir}_{TensorboardModelLog.add_summary.__name__}_{id(model)}'

        model_str, nn_model_summary = get_model_summaries(
            model=model,
            input_size=input_size,
            train_loader=train_loader,
            *args,
            **kwargs
        )

        if log_id in self._log_record:
            return model_str, nn_model_summary

        self.tensorboard.add_text(
            f'str ({model.__class__.__name__})',
            re.sub('\n', '\n    ', f'    {model_str}')
        )
        self.tensorboard.add_text(
            f'summary ({model.__class__.__name__})',
            re.sub('\n', '\n    ', f'    {nn_model_summary}')
        )
        self._log_record[log_id] = True
        return model_str, nn_model_summary

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
        self.tensorboard.add_scalar('Accuracy/train', train_accuracy, epoch)
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
        self.tensorboard.add_scalar('Accuracy/test', test_accuracy, epoch)
        return self

    # noinspection PyUnusedLocal
    def close(self, *args, **kwargs):
        """Close the tensorboard.

        Args:
            *args: ignored.
            **kwargs: ignored.
        """

        if self.tensorboard is not None:
            self.tensorboard.close()
            self.tensorboard = None

    def __enter__(self):
        """Enter the TensorboardModelLog context.

        Returns:
            TensorboardModelLog: self.
        """

        return self

    __exit__ = close
    """Close the tensorboard."""

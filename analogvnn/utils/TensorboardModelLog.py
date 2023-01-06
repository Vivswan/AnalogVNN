import os
import re

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from analogvnn.utils.summary import summary

__all__ = ['TensorboardModelLog']


class TensorboardModelLog:
    __constants__ = ['model', 'log_dir']
    log_dir: str

    def __init__(self, model, log_dir: str):
        self.model = model
        self.tensorboard = None
        self.layer_data = True

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.set_log_dir(log_dir)
        if hasattr(model, "subscribe_tensorboard"):
            model.subscribe_tensorboard(tensorboard=self)

    def set_log_dir(self, log_dir: str):
        if os.path.isdir(log_dir):
            self.tensorboard = SummaryWriter(log_dir=log_dir)
        else:
            raise Exception(f'"{log_dir}" is not a directory.')

    def _add_layer_data(self, epoch: int = None):
        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList) or (module == self):
                continue

            idx += 1
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                self.tensorboard.add_histogram(f"{idx}-{module}.bias", module.bias, epoch)
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                self.tensorboard.add_histogram(f"{idx}-{module}.weight", module.weight, epoch)

    def on_compile(self, layer_data=True):
        if self.layer_data:
            self.layer_data = layer_data

        if self.layer_data:
            self._add_layer_data(epoch=-1)
        return self

    def add_graph(self, train_loader, model=None):
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

    def register_training(self, epoch, train_loss, train_accuracy):
        self.tensorboard.add_scalar('Loss/train', train_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/train", train_accuracy, epoch)
        if self.layer_data:
            self._add_layer_data(epoch=epoch)

    def register_testing(self, epoch, test_loss, test_accuracy):
        self.tensorboard.add_scalar('Loss/test', test_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/test", test_accuracy, epoch)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

import os
import re

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from nn.utils.summary import summary


class TensorboardModelLog:
    __constants__ = ['model', 'log_dir']
    log_dir: str

    def __init__(self, model, log_dir: str):
        self.model = model
        self.tensorboard = None
        self.set_log_dir(log_dir)
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
            if hasattr(module, "random_weight") and hasattr(module.random_weight, "size"):
                self.tensorboard.add_histogram(f"{idx}-{module}.weight", module.random_weight, epoch)

    def on_compile(self):
        self.tensorboard.add_text("summary", re.sub("\n", "\n    ", "    " + summary(self.model)))
        self.tensorboard.add_text("str", re.sub("\n", "\n    ", "    " + str(self.model)))
        self.tensorboard.add_graph(self.model, torch.zeros(tuple([1] + list(self.model.in_features))).to(self.model.device))

        self._add_layer_data(epoch=-1)
        return self

    def register_training(self, epoch, train_loss, train_accuracy):
        self.tensorboard.add_scalar('Loss/train', train_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/train", train_accuracy, epoch)
        self._add_layer_data(epoch=epoch)

    def register_testing(self, epoch, test_loss, test_accuracy):
        self.tensorboard.add_scalar('Loss/test', test_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/test", test_accuracy, epoch)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


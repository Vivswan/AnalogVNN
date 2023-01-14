from __future__ import annotations

from typing import TypeVar, Optional

import torch
from torch import nn

from analogvnn.nn.module.Model import Model

T = TypeVar('T', bound=nn.Module)

__all__ = ['Sequential']


class Sequential(Model, nn.Sequential):
    """Base class for all sequential models."""

    def __call__(self, *args, **kwargs):
        """Call the model.

        Args:
            *args: The input.
            **kwargs: The input.

        Returns:
            torch.Tensor: The output of the model.
        """

        if not self._compiled:
            self.compile()

        return super().__call__(*args, **kwargs)

    def compile(self, device: Optional[torch.device] = None, layer_data: bool = True):
        """Compile the model and add forward graph.

        Args:
            device (torch.device): The device to run the model on.
            layer_data (bool): True if the data of the layers should be compiled.

        Returns:
            Sequential: self
        """
        arr = [self.graphs.INPUT, *list(self.registered_children()), self.graphs.OUTPUT]
        self.graphs.forward_graph.add_connection(*arr)
        return super().compile(device, layer_data)

    def add_sequence(self, *args):
        """Add a sequence of modules to the forward graph of model.

        Args:
            *args (nn.Module): The modules to add.
        """

        return self.extend(args)

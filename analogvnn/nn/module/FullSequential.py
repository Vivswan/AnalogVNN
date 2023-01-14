from __future__ import annotations

from typing import Optional

import torch

from analogvnn.nn.module.Sequential import Sequential

__all__ = ['FullSequential']


class FullSequential(Sequential):
    """A sequential model where backward graph is the reverse of forward graph."""

    def compile(self, device: Optional[torch.device] = None, layer_data: bool = True):
        """Compile the model and add forward and backward graph.

        Args:
            device (torch.device): The device to run the model on.
            layer_data (bool): True if the data of the layers should be compiled.

        Returns:
            FullSequential: self
        """
        arr = [self.graphs.INPUT, *list(self.registered_modules()), self.graphs.OUTPUT]
        self.graphs.backward_graph.add_connection(*reversed(arr))
        return super().compile(device, layer_data)

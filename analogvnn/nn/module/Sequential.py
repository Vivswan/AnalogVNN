from __future__ import annotations

import operator
from collections import OrderedDict
from itertools import islice
from typing import Iterator, TypeVar, Union, Dict, Optional

import torch
from torch import nn
from torch.nn import Module

from analogvnn.nn.module.Model import Model

T = TypeVar('T', bound=nn.Module)

__all__ = ['Sequential']


class Sequential(Model):
    """A sequential model.

    Attributes:
        _runtime_module_list (OrderedDict[str, nn.Module]): The ordered dictionary of the modules.
    """

    def __init__(self, *args):
        """Initialize the model.

        Args:
            *args: The modules to add.
        """
        super(Sequential, self).__init__()
        self._runtime_module_list: Dict[str, Optional[Module]] = OrderedDict()
        self.add_sequence(*args)

    def compile(self, device: Optional[torch.device] = None, layer_data: bool = True):
        """Compile the model and add forward graph.

        Args:
            device (torch.device): The device to run the model on.
            layer_data (bool): True if the data of the layers should be compiled.

        Returns:
            Sequential: self
        """
        arr = [self.graphs.INPUT, *list(self._runtime_module_list.values()), self.graphs.OUTPUT]
        self.graphs.forward_graph.add_connection(*arr)
        return super().compile(device, layer_data)

    def add_sequence(self, *args):
        """Add a sequence of modules to the forward graph of model.

        Args:
            *args (nn.Module): The modules to add.
        """
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._add_run_module(key, module)
        else:
            for idx, module in enumerate(args):
                self._add_run_module(str(idx), module)

    def _add_run_module(self, name: str, module: Optional[Module]):
        """Add a module to the forward graph of model.

        Args:
            name (str): The name of the module.
            module (nn.Module): The module to add.
        """
        self.add_module(name, module)
        self._runtime_module_list[name] = module
        return self

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator.

        Args:
            iterator (Iterator): The iterator.
            idx (int): The index of the item.

        Returns:
            T: The idx-th item of the iterator.
        """
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union[Sequential, T]:
        """Get the idx-th module of the model.

        Args:
            idx (int): The index of the module.

        Returns:
            Union[Sequential, T]: The idx-th module of the model.
        """
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._runtime_module_list.items())[idx]))
        else:
            return self._get_item_by_idx(self._runtime_module_list.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        """Set the idx-th module of the model.

        Args:
            idx (int): The index of the module.
            module (nn.Module): The module to set.
        """
        key: str = self._get_item_by_idx(self._runtime_module_list.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        """Remove the idx-th module from the model.

        Args:
            idx (Union[slice, int]): The index of the module.
        """
        if isinstance(idx, slice):
            for key in list(self._runtime_module_list.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._runtime_module_list.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        """Return the number of modules in the model.

        Returns:
            int: The number of modules in the model.
        """
        return len(self._runtime_module_list)

    def __dir__(self):
        """Return the list of attributes of the module.

        Returns:
            list: The list of attributes of the module.
        """
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[nn.Module]:
        """Return an iterator over the modules of the model.

        Returns:
            Iterator[nn.Module]: An iterator over the modules of the model.
        """
        return iter(self._runtime_module_list.values())

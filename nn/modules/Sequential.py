import operator
from collections import OrderedDict
from itertools import islice
from typing import Iterator, TypeVar, Union, Dict, Optional

import torch
from torch import nn
from torch.nn import Module

from nn.modules.BaseModule import BaseModule

T = TypeVar('T', bound=nn.Module)


class Sequential(BaseModule):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._runtime_module_list: Dict[str, Optional[Module]] = OrderedDict()
        self.add_sequence(*args)

    def add_sequence(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._add_run_module(key, module)
        else:
            for idx, module in enumerate(args):
                self._add_run_module(str(idx), module)

    def _add_run_module(self, name: str, module: Optional[Module]):
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._runtime_module_list[name] = module
        self._modules[name] = module
        return self

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def forward(self, x):
        for module in self._runtime_module_list.values():
            x = module(x)
        return x

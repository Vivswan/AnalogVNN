import operator
from collections import OrderedDict
from itertools import islice
from typing import Iterator, TypeVar, Union, Dict, Optional

from torch import nn
from torch.nn import Module

from nn.modules.Model import Model

T = TypeVar('T', bound=nn.Module)


class Sequential(Model):
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
        self.add_module(name, module)
        self._runtime_module_list[name] = module
        return self

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index.rst {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._runtime_module_list.items())[idx]))
        else:
            return self._get_item_by_idx(self._runtime_module_list.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key: str = self._get_item_by_idx(self._runtime_module_list.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._runtime_module_list.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._runtime_module_list.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._runtime_module_list)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._runtime_module_list.values())

    def forward(self, x):
        for module in self._runtime_module_list.values():
            x = module(x)
        return x

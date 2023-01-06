from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

__all__ = ['InputOutput', 'ArgsKwargs']


@dataclass
class InputOutput:
    inputs: Optional[ArgsKwargs] = None
    outputs: Optional[ArgsKwargs] = None


@dataclass
class ArgsKwargs:
    args: List
    kwargs: Dict

    def __init__(self, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        if isinstance(args, tuple):
            args = list(args)

        if not isinstance(args, List):
            args = [args]

        self.args = args
        self.kwargs = kwargs

    def is_empty(self):
        return len(self.args) == 0 and not bool(self.kwargs)

    def __repr__(self):
        return f"ArgsKwargs(args={self.args}, kwargs={self.kwargs})"

    @staticmethod
    def to_args_kwargs_object(outputs) -> ArgsKwargs:
        if isinstance(outputs, ArgsKwargs):
            pass
        elif isinstance(outputs, dict):
            outputs = ArgsKwargs(kwargs=outputs)
        elif isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
            outputs = ArgsKwargs(args=outputs[0], kwargs=outputs[1])
        else:
            outputs = ArgsKwargs(args=outputs)
        return outputs

    @staticmethod
    def from_args_kwargs_object(outputs) -> Union[ArgsKwargs, List, Any, None]:
        if len(outputs.kwargs.keys()) > 0:
            return outputs
        elif len(outputs.args) > 1:
            return outputs.args
        elif len(outputs.args) == 1:
            return outputs.args[0]
        else:
            return None

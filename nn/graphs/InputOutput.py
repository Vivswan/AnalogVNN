from dataclasses import dataclass
from typing import Tuple, Union, List, Dict

from torch import Tensor


@dataclass
class InputOutput:
    def __init__(self, inputs=None, outputs=None):
        self.inputs: ArgsKwargs = inputs
        self.outputs: ArgsKwargs = outputs

    def __repr__(self):
        return f"InputOutput(inputs={self.inputs}, outputs={self.outputs})"


@dataclass
class ArgsKwargs:
    def __init__(self, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        if isinstance(args, tuple):
            args = list(args)

        if not isinstance(args, List):
            args = [args]

        self.args: List = args
        self.kwargs: Dict = kwargs

    def is_empty(self):
        return len(self.args) == 0 and not bool(self.kwargs)

    def __repr__(self):
        return f"ArgsKwargs(args={self.args}, kwargs={self.kwargs})"


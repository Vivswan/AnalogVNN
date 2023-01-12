from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

__all__ = ['InputOutput', 'ArgsKwargs', 'ArgsKwargsInput', 'ArgsKwargsOutput']


@dataclass
class InputOutput:
    """Inputs and outputs of a module.

    Attributes:
        inputs (Optional[ArgsKwargs]): Inputs of a module.
        outputs (Optional[ArgsKwargs]): Outputs of a module.
    """
    inputs: Optional[ArgsKwargs] = None
    outputs: Optional[ArgsKwargs] = None


@dataclass
class ArgsKwargs:
    """The arguments.

    Attributes:
        args (List): The arguments.
        kwargs (Dict): The keyword arguments.
    """
    args: List
    kwargs: Dict

    def __init__(self, args=None, kwargs=None):
        """Initialize the ArgsKwargs object.

        Args:
            args: The arguments.
            kwargs: The keyword arguments.
        """
        super(ArgsKwargs, self).__init__()
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
        """Returns whether the ArgsKwargs object is empty."""
        return len(self.args) == 0 and not bool(self.kwargs)

    def __repr__(self):
        """Returns a string representation of the parameter."""
        return f"ArgsKwargs(args={self.args}, kwargs={self.kwargs})"

    @classmethod
    def to_args_kwargs_object(cls, outputs: ArgsKwargsInput) -> ArgsKwargs:
        """ Convert the output of a module to ArgsKwargs object

        Args:
            outputs: The output of a module

        Returns:
            ArgsKwargs: The ArgsKwargs object
        """
        if isinstance(outputs, cls):
            pass
        elif isinstance(outputs, dict):
            outputs = cls(kwargs=outputs)
        elif isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
            outputs = cls(args=outputs[0], kwargs=outputs[1])
        else:
            outputs = cls(args=outputs)
        return outputs

    @staticmethod
    def from_args_kwargs_object(outputs: ArgsKwargs) -> ArgsKwargsOutput:
        """Convert ArgsKwargs to object or tuple or dict

        Args:
            outputs (ArgsKwargs): ArgsKwargs object

        Returns:
            ArgsKwargsOutput: object or tuple or dict
        """
        if len(outputs.kwargs.keys()) > 0:
            return outputs
        elif len(outputs.args) > 1:
            return outputs.args
        elif len(outputs.args) == 1:
            return outputs.args[0]
        else:
            return None


ArgsKwargsInput = Union[ArgsKwargs, Dict, List, Any, None]
"""ArgsKwargsInput is the input type for ArgsKwargs"""

ArgsKwargsOutput = Union[ArgsKwargs, List, Any, None]
"""ArgsKwargsOutput is the output type for ArgsKwargs"""

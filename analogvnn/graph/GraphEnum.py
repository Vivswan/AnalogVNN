from __future__ import annotations

import enum
from typing import Union, Callable

__all__ = ['GraphEnum', 'GRAPH_NODE_TYPE']


class GraphEnum(enum.Enum):
    """The graph enum for indicating input, output and stop.

    Attributes:
        INPUT (GraphEnum): GraphEnum.INPUT
        OUTPUT (GraphEnum): GraphEnum.OUTPUT
        STOP (GraphEnum): GraphEnum.STOP
    """

    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    STOP = 'STOP'


GRAPH_NODE_TYPE = Union[GraphEnum, Callable]

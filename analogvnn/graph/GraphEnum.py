import enum

__all__ = ['GraphEnum']


class GraphEnum(enum.Enum):
    """The graph enum for indicating input, output and stop.

    Attributes:
        INPUT (GraphEnum): GraphEnum.INPUT
        OUTPUT (GraphEnum): GraphEnum.OUTPUT
        STOP (GraphEnum): GraphEnum.STOP
    """
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    STOP = "STOP"

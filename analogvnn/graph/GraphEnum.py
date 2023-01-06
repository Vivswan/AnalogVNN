import enum

__all__ = ['GraphEnum']


class GraphEnum(enum.Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    STOP = "STOP"

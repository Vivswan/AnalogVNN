from torch import nn

__all__ = ['Parameter']


class Parameter(nn.Parameter):
    """A parameter that can be used in a `torch.nn.Module`.

    This class is a wrapper around `torch.nn.Parameter` that allows to set additional attributes.
    """

    def __new__(cls, data=None, requires_grad=True, *args, **kwargs):
        """Creates a new parameter.

        Args:
            data: the data for the parameter.
            requires_grad (bool): whether the parameter requires gradient.
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            Parameter: the created parameter.
        """
        return super(Parameter, cls).__new__(cls, data, requires_grad)

    # noinspection PyUnusedLocal
    def __init__(self, data=None, requires_grad=True, *args, **kwargs):
        """Initializes the parameter.

        Args:
            data: the data for the parameter.
            requires_grad (bool): whether the parameter requires gradient.
            *args: additional arguments.
            **kwargs: additional keyword arguments.
        """
        super(Parameter, self).__init__()

    def __repr__(self, *args, **kwargs):
        """Returns a string representation of the parameter.

        Args:
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            str: the string representation.
        """
        return super(Parameter, self).__repr__(*args, **kwargs)

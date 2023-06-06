from typing import Optional, Sequence, Tuple

from torch import nn
from torch.utils.data import DataLoader

from analogvnn.nn.module.Layer import Layer


def get_model_summaries(  # noqa: C901
        model: Optional[nn.Module],
        input_size: Optional[Sequence[int]] = None,
        train_loader: DataLoader = None,
        *args,
        **kwargs
) -> Tuple[str, str]:
    """Creates the model summaries.

    Args:
        train_loader (DataLoader): the train loader.
        model (nn.Module): the model to log.
        input_size (Optional[Sequence[int]]): the input size.
        *args: the arguments to torchinfo.summary.
        **kwargs: the keyword arguments to torchinfo.summary.

    Returns:
        Tuple[str, str]: the model __repr__ and the model summary.

    Raises:
        ImportError: if torchinfo (https://github.com/tyleryep/torchinfo) is not installed.
        ValueError: if the input_size and train_loader are None.
    """

    try:
        import torchinfo
    except ImportError as e:
        raise ImportError('requires torchinfo: https://github.com/tyleryep/torchinfo') from e

    if input_size is None and train_loader is None and 'input_size' not in kwargs:
        raise ValueError('input_size or train_loader must be provided')

    if 'input_size' not in kwargs:
        if input_size is None:
            data_shape = list(next(iter(train_loader))[0].shape)
            if train_loader.batch_size > 0:
                data_shape[0] = 1

            input_size = data_shape

        kwargs['input_size'] = input_size

    use_autograd_graph = False
    if isinstance(model, Layer):
        use_autograd_graph = model.use_autograd_graph
        model.use_autograd_graph = True

    if 'depth' not in kwargs:
        kwargs['depth'] = 10
    if 'col_names' not in kwargs:
        kwargs['col_names'] = tuple(e.value for e in torchinfo.ColumnSettings)
    if 'verbose' not in kwargs:
        kwargs['verbose'] = torchinfo.Verbosity.QUIET

    model_summary = torchinfo.summary(
        model,
        *args,
        **kwargs,
    )

    if isinstance(model, Layer):
        model.use_autograd_graph = use_autograd_graph

    model_summary.formatting.verbose = torchinfo.Verbosity.VERBOSE
    model_str = str(model)
    model_summary = f'{model_summary}'
    return model_str, model_summary

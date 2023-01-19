from typing import Optional, Sequence, Tuple

from torch import nn
from torch.utils.data import DataLoader

from analogvnn.nn.module.Layer import Layer


def get_model_summaries(
        model: Optional[nn.Module],
        input_size: Optional[Sequence[int]] = None,
        train_loader: DataLoader = None,
) -> Tuple[str, str]:
    """Creates the model summaries.

    Args:
        train_loader (DataLoader): the train loader.
        model (nn.Module): the model to log.
        input_size (Optional[Sequence[int]]): the input size.

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

    if input_size is None and train_loader is None:
        raise ValueError('input_size or train_loader must be provided')

    if input_size is None:
        data_shape = next(iter(train_loader))[0].shape
        input_size = tuple(list(data_shape)[1:])

    use_autograd_graph = False
    if isinstance(model, Layer):
        use_autograd_graph = model.use_autograd_graph
        model.use_autograd_graph = True

    model_summary = torchinfo.summary(
        model,
        input_size=input_size,
        verbose=torchinfo.Verbosity.QUIET,
        col_names=[e.value for e in torchinfo.ColumnSettings],
        depth=10,
    )

    if isinstance(model, Layer):
        model.use_autograd_graph = use_autograd_graph

    model_summary.formatting.verbose = torchinfo.Verbosity.VERBOSE
    model_str = str(model)
    model_summary = f'{model_summary}'
    return model_str, model_summary

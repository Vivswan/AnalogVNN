import sys

if sys.version_info[:2] >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata  # pragma: no cover


__package__ = 'analogvnn'
__author__ = 'Vivswan Shah (vivswanshah@pitt.edu)'

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

if sys.version_info < (3, 7, 0):
    import warnings

    warnings.warn(
        'The installed Python version reached its end-of-life. '
        'Please upgrade to a newer Python version for receiving '
        'further gdshelpers updates.', Warning)

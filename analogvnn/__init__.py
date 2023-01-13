import sys

import importlib_metadata

__package__ = 'analogvnn'
__author__ = 'Vivswan Shah (vivswanshah@pitt.edu)'

try:
    __version__ = importlib_metadata.version(__package__)
except importlib_metadata.PackageNotFoundError:
    __version__ = 'local'

if sys.version_info < (3, 7, 0):
    import warnings

    warnings.warn(
        'The installed Python version reached its end-of-life. '
        'Please upgrade to a newer Python version for receiving '
        'further gdshelpers updates.', Warning)

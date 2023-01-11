import sys

import importlib_metadata

__package__ = "analogvnn"
__version__ = importlib_metadata.version(__package__)
__author__ = "Vivswan Shah (vivswanshah@pitt.edu)"

if sys.version_info < (3, 7, 0):
    import warnings

    warnings.warn(
        'The installed Python version reached its end-of-life. '
        'Please upgrade to a newer Python version for receiving '
        'further gdshelpers updates.', Warning)

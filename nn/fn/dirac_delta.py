import numpy as np


def dirac_delta(x, a=0.001):
    return 1 / (np.abs(a) * np.sqrt(np.pi)) * np.exp(-((x / a) ** 2))

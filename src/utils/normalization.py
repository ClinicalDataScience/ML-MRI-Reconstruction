"""Functions for normalization."""
import numpy as np


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """Apply min-max-normalization."""
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))

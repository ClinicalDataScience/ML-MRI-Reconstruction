"""Functions for normalization."""
import numpy as np


def root_sum_of_squares(reconstruction: np.ndarray, axis_num: int) -> np.ndarray:
    """Apply a root-sum-of-squares algorithm."""
    reconstruction = np.sum(reconstruction**2, axis=axis_num)
    reconstruction = np.sqrt(reconstruction)
    return reconstruction

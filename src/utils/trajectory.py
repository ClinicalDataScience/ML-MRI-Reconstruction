"""Function for defining a radial k-space trajectory."""
import logging
import sys

import numpy as np


def define_radial_trajectory(num_spokes: int, num_readouts: int) -> np.ndarray:
    """Define radial k-space trajectory for an odd number of spokes."""
    if (num_spokes % 2) == 0:
        logging.warning('This function requires an odd number of spokes.')
        sys.exit('This function requires an odd number of spokes.')
    traj = np.zeros((num_spokes * num_readouts, 2))
    vec = np.pi * np.linspace(1, -1, num_readouts, endpoint=False)
    for n in range(num_spokes):
        phi = 2 * np.pi * (n / num_spokes)
        traj[n * num_readouts : (n + 1) * num_readouts, 0] = vec * np.cos(phi)
        traj[n * num_readouts : (n + 1) * num_readouts, 1] = vec * np.sin(phi)
    return traj

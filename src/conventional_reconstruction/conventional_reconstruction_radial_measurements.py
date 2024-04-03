"""Functions for the conventional reconstruction of radial k-space measurements."""
import os
import subprocess
from shlex import split
from typing import Optional, Union

import numpy as np
from src.utils.mathematical_operations import root_sum_of_squares
from src.utils.save_and_load import readcfl, writecfl


def apply_density_correction_for_nufft_adjoint(
    path_to_k_space_old: Union[str, os.PathLike],
    path_to_k_space_new: Union[str, os.PathLike],
    dens_correction_nufft_adjoint: np.ndarray,
) -> None:
    """Apply density correction to k-space input."""
    k_space = readcfl(path_to_k_space_old)
    k_space_dens_corr = k_space * dens_correction_nufft_adjoint
    writecfl(path_to_k_space_new, k_space_dens_corr)


def postprocessing_bart_measurements(
    method: str, path_to_recon_image: Union[str, os.PathLike]
) -> np.ndarray:
    """Postprocess image with BART."""
    # load reconstructed image for post-processing
    reconstruction = readcfl(path_to_recon_image)
    # combine reconstructed coil images with a root-sum-of-squares algorithm
    if method != 'CS':
        reconstruction = root_sum_of_squares(reconstruction, -1)
    # calculate magnitude image
    reconstruction = np.abs(reconstruction)
    reconstruction = np.squeeze(reconstruction)
    return reconstruction


def reconstruct_k_space_measurements_with_bart(
    cmp: str,
    method: str,
    path_to_recon_image: Union[str, os.PathLike],
    path_to_k_space_old: Optional[Union[str, os.PathLike]] = None,
    path_to_k_space_new: Optional[Union[str, os.PathLike]] = None,
    dens_correction_nufft_adjoint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct image with BART."""
    if (
        method == 'nufft_adjoint'
        and path_to_k_space_old is not None
        and path_to_k_space_new is not None
        and dens_correction_nufft_adjoint is not None
    ):
        apply_density_correction_for_nufft_adjoint(
            path_to_k_space_old, path_to_k_space_new, dens_correction_nufft_adjoint
        )
    subprocess.run(split(cmp), shell=False, check=True)
    reconstruction = postprocessing_bart_measurements(method, path_to_recon_image)
    return reconstruction

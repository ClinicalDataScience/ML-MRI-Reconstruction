"""Functions for the conventional reconstruction of synthetic k-space data."""
import os
import subprocess
from shlex import split
from typing import Optional, Union

import numpy as np
from src.utils.save_and_load import readcfl, writecfl


def reshape_input_for_bart_reconstruction(
    X: np.ndarray, num_spokes: int, num_readouts: int, im_w: int
) -> None:
    """Reshape input for the BART reconstruction."""
    X = np.reshape(X, (num_spokes, num_readouts, 1))
    X = np.transpose(X, (2, 1, 0))
    X = X / im_w
    return X


def scale_nufft_adjoint(
    reconstruction: np.ndarray, num_spokes: int, im_w: int
) -> np.ndarray:
    """Scale reconstruction after adjoint NUFFT."""
    us = (im_w / 2 * np.pi) / num_spokes
    reconstruction = (
        reconstruction / (im_w * 4 * (np.pi / 2)) * us * 128
    )  # normalization factor for im_w = 128
    return reconstruction


def apply_density_correction_for_nufft_adjoint(
    path_to_k_space: Union[str, os.PathLike], dens_correction_nufft_adjoint: np.ndarray
) -> None:
    """Apply density correction to k-space input."""
    k_space = readcfl(path_to_k_space)
    k_space_dens_corr = k_space * dens_correction_nufft_adjoint
    writecfl(path_to_k_space, k_space_dens_corr)


def postprocessing_bart_single_image(
    method: str,
    path_to_recon_image: Union[str, os.PathLike],
    num_spokes: Optional[int] = None,
    im_w: Optional[int] = None,
) -> np.ndarray:
    """Postprocess image after reconstruction with BART."""
    reconstruction = readcfl(path_to_recon_image)
    if method == 'nufft_adjoint' and num_spokes and im_w:
        reconstruction = scale_nufft_adjoint(reconstruction, num_spokes, im_w)
    reconstruction = np.abs(reconstruction)
    return reconstruction


def reconstruct_synthetic_k_space_data_with_bart(
    cmp: str,
    method: str,
    path_to_recon_image: Union[str, os.PathLike],
    num_spokes: Optional[int] = None,
    im_w: Optional[int] = None,
    path_to_k_space: Optional[Union[str, os.PathLike]] = None,
    dens_correction_nufft_adjoint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct image with BART."""
    if (
        method == 'nufft_adjoint'
        and path_to_k_space is not None
        and dens_correction_nufft_adjoint is not None
        and im_w is not None
    ):
        apply_density_correction_for_nufft_adjoint(
            path_to_k_space, dens_correction_nufft_adjoint
        )
    subprocess.run(split(cmp), shell=False, check=True)
    reconstruction = postprocessing_bart_single_image(
        method, path_to_recon_image, num_spokes, im_w
    )
    return reconstruction

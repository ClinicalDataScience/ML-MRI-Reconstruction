"""Functions for the conventional reconstruction of radial k-space measurements."""
import logging
import os
import subprocess
import sys
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


def calculate_coil_sensitivity_maps_for_compressed_sensing(
    path_traj_bart: Union[str, os.PathLike],
    path_to_kspace_data: Union[str, os.PathLike],
    im_w: int,
    tmpdirname: Union[str, os.PathLike],
) -> None:
    """Calculate coil sensitivity maps for compressed sensing."""
    # reconstruct low-resolution image with inverse NUFFT
    path_to_low_res_nufft = os.path.join(tmpdirname, 'low_resolution_image_after_nufft')
    cmp_nufft_low_res = f'bart nufft -i -d 24:24:1 -t {path_traj_bart} {path_to_kspace_data} {path_to_low_res_nufft}'
    subprocess.run(split(cmp_nufft_low_res), shell=False, check=True)

    # transform low-resolution image back to k-space
    path_to_lowres_ksp = os.path.join(tmpdirname, 'lowresolution_kspace')
    cmp_lowres_ksp = f'bart fft -u 7 {path_to_low_res_nufft} {path_to_lowres_ksp}'
    subprocess.run(split(cmp_lowres_ksp), shell=False, check=True)

    # zeropad to full size
    path_to_ksp_zerop = os.path.join(tmpdirname, 'kspace_zeropad')
    cmp_ksp_zerop = (
        f'bart resize -c 0 {im_w} 1 {im_w} {path_to_lowres_ksp} {path_to_ksp_zerop}'
    )
    subprocess.run(split(cmp_ksp_zerop), shell=False, check=True)

    # ESPIRiT calibration
    path_to_espirit_calibration = os.path.join(tmpdirname, 'espirit_calibration')
    cmp_espirit_calibration = (
        f'bart ecalib -m1 {path_to_ksp_zerop} {path_to_espirit_calibration}'
    )
    subprocess.run(split(cmp_espirit_calibration), shell=False, check=True)


def postprocessing_bart_measurements(
    method: str, path_to_recon_image: Union[str, os.PathLike]
) -> np.ndarray:
    """Postprocess image for BART reconstruction."""
    # load reconstructed image for post-processing
    reconstruction = readcfl(path_to_recon_image)
    # calculate magnitude image
    reconstruction = np.abs(reconstruction)
    # combine reconstructed coil images with a root-sum-of-squares algorithm
    if method != 'CS':
        reconstruction = root_sum_of_squares(reconstruction, -1)
    reconstruction = np.squeeze(reconstruction)
    return reconstruction


def reconstruct_k_space_measurements_with_nufft_adjoint(
    cmp: str,
    path_to_recon_image: Union[str, os.PathLike],
    path_to_k_space_old: Union[str, os.PathLike],
    path_to_k_space_new: Union[str, os.PathLike],
    dens_correction_nufft_adjoint: np.ndarray,
) -> np.ndarray:
    """Reconstruct k-space measurement with NUFFT."""
    apply_density_correction_for_nufft_adjoint(
        path_to_k_space_old, path_to_k_space_new, dens_correction_nufft_adjoint
    )
    subprocess.run(split(cmp), shell=False, check=True)
    reconstruction = postprocessing_bart_measurements(
        'nufft_adjoint', path_to_recon_image
    )
    return reconstruction


def reconstruct_k_space_measurements_with_compressed_sensing(
    cmp: str,
    path_to_recon_image: Union[str, os.PathLike],
    path_to_k_space_old: Union[str, os.PathLike],
    im_w: int,
    path_traj_bart: np.ndarray,
    tmpdirname: np.ndarray,
) -> np.ndarray:
    """Reconstruct k-space measurement with compressed sensing."""
    calculate_coil_sensitivity_maps_for_compressed_sensing(
        path_traj_bart, path_to_k_space_old, im_w, tmpdirname
    )
    subprocess.run(split(cmp), shell=False, check=True)
    reconstruction = postprocessing_bart_measurements('CS', path_to_recon_image)
    return reconstruction

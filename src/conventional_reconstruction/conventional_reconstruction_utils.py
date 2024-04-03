"""Functions for setting up the BART reconstruction."""
import os
import subprocess
from typing import Optional, Union

import numpy as np
from src.utils.save_and_load import writecfl
from src.utils.trajectory import define_radial_trajectory


def define_trajectory_bart(
    num_spokes: int,
    num_readouts: int,
) -> np.ndarray:
    """Reshape radial trajectory to the shape that is required by BART."""
    traj = define_radial_trajectory(num_spokes, num_readouts)
    traj_bart = np.zeros((traj.shape[0], 3), dtype=traj.dtype)
    traj_bart[:, :2] = traj / np.pi * (num_readouts // 4)
    traj_bart = traj_bart.reshape((num_spokes, num_readouts, 3))
    traj_bart = traj_bart.transpose((2, 1, 0))
    return traj_bart


def save_radial_trajectory_bart(
    num_spokes: int, num_readouts: int, path_to_radial_trajectory_bart: str
) -> None:
    """Save the radial trajectory for the BART reconstruction."""
    traj_bart = define_trajectory_bart(num_spokes, num_readouts)
    writecfl(path_to_radial_trajectory_bart, traj_bart)


def save_sensitivity_map_bart(
    im_w: int, path_to_sensitivity_map: Union[str, os.PathLike]
) -> None:
    """Save the sensitivity map for the BART reconstruction."""
    cmp = f'bart ones 4 {im_w} {im_w} 1 1 {path_to_sensitivity_map}'
    subprocess.run(cmp, shell=True, check=True)


def save_sensitivity_map_bart_coils(
    im_w: int, num_coils: int, path_to_sensitivity_map: Union[str, os.PathLike]
) -> None:
    """Save the sensitivity map for the BART reconstruction."""
    cmp = f'bart ones 4 {im_w} {im_w} 1 {num_coils} {path_to_sensitivity_map}'
    subprocess.run(cmp, shell=True, check=True)


def return_command_for_nufft_reconstruction_bart(
    method: str,
    path_traj_bart: Union[str, os.PathLike],
    path_to_save_kspace: Union[str, os.PathLike],
    path_to_recon_image: Union[str, os.PathLike],
    device: str,
) -> str:
    """Return the command for the NUFFT BART reconstruction."""
    if method == 'nufft_adjoint':
        cmp = f'bart nufft -a {path_traj_bart} {path_to_save_kspace} {path_to_recon_image}'
    elif method == 'nufft_inverse':
        cmp = f'bart nufft -i {path_traj_bart} {path_to_save_kspace} {path_to_recon_image}'

    # if device is cuda: add -g flag
    if device == 'cuda':
        if method == 'nufft_adjoint':
            substr = '-a'
        elif method == 'nufft_inverse':
            substr = '-i'
        inserttxt = '-g '
        idx = cmp.index(substr)
        cmp = cmp[:idx] + inserttxt + cmp[idx:]

    return cmp


def return_command_for_compressed_sensing_reconstruction_bart(
    path_traj_bart: Union[str, os.PathLike],
    path_to_save_kspace: Union[str, os.PathLike],
    path_to_recon_image: Union[str, os.PathLike],
    path_to_sensitivity_map: Union[str, os.PathLike],
    bart_maxiter: int,
    bart_regularization_option: str,
    bart_regularization: float,
    device: str,
    bart_bitmask: Optional[int] = None,
    stepsize: Optional[float] = None,
) -> str:
    """Return the command for the compressed sensing BART reconstruction."""
    if bart_bitmask:
        if (
            bart_regularization_option == 'W'
            or bart_regularization_option == 'N'
            or bart_regularization_option == 'H'
            or bart_regularization_option == 'F'
            or bart_regularization_option == 'T'
            or bart_regularization_option == 'L'
        ):
            cmp = f'bart pics -S -R {bart_regularization_option}:{bart_bitmask}:0:{bart_regularization} -i {bart_maxiter} -t {path_traj_bart} {path_to_save_kspace} {path_to_sensitivity_map} {path_to_recon_image}'
    else:
        if bart_regularization_option == 'Q':
            cmp = f'bart pics -S -R {bart_regularization_option}:{bart_regularization} -i {bart_maxiter} -t {path_traj_bart} {path_to_save_kspace} {path_to_sensitivity_map} {path_to_recon_image}'
        elif bart_regularization_option == 'I':
            cmp = f'bart pics -S -R {bart_regularization_option}:0:{bart_regularization} -i {bart_maxiter} -t {path_traj_bart} {path_to_save_kspace} {path_to_sensitivity_map} {path_to_recon_image}'
        elif bart_regularization_option == 'l1':
            cmp = f'bart pics -S -l1 -r {bart_regularization} -i {bart_maxiter} -t {path_traj_bart} {path_to_save_kspace} {path_to_sensitivity_map} {path_to_recon_image}'

    # if stepsize is defines use this stepsize; otherwise use -e flag
    if stepsize:
        substr = '-S'
        inserttxt = f'-s {stepsize} '
        idx = cmp.index(substr)
        cmp = cmp[:idx] + inserttxt + cmp[idx:]
    else:
        substr = '-S'
        inserttxt = f'-e '
        idx = cmp.index(substr)
        cmp = cmp[:idx] + inserttxt + cmp[idx:]

    # if device is cuda: add -g flag
    if device == 'cuda':
        substr = '-S'
        inserttxt = f'-g '
        idx = cmp.index(substr)
        cmp = cmp[:idx] + inserttxt + cmp[idx:]

    return cmp


def return_command_for_bart_reconstruction(
    method: str,
    path_traj_bart: str,
    path_to_save_kspace: str,
    path_to_recon_image: str,
    device: str,
    path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
    bart_maxiter: Optional[int] = None,
    bart_regularization_option: Optional[str] = None,
    bart_regularization: Optional[float] = None,
    bart_bitmask: Optional[int] = None,
    stepsize: Optional[float] = None,
) -> str:
    """Return the command for BART reconstruction."""
    if method == 'CS':
        if (
            path_to_sensitivity_map
            and bart_maxiter
            and bart_regularization_option
            and bart_regularization
        ):
            cmp = return_command_for_compressed_sensing_reconstruction_bart(
                path_traj_bart,
                path_to_save_kspace,
                path_to_recon_image,
                path_to_sensitivity_map,
                bart_maxiter,
                bart_regularization_option,
                bart_regularization,
                device,
                bart_bitmask,
                stepsize,
            )
    elif method == 'nufft_adjoint' or method == 'nufft_inverse':
        cmp = return_command_for_nufft_reconstruction_bart(
            method, path_traj_bart, path_to_save_kspace, path_to_recon_image, device
        )
    return cmp


def warm_up_bart(
    num_spokes: int,
    num_readouts: int,
    cmp: str,
    path_to_save_kspace: Union[str, os.PathLike],
    device: str,
    num_coils: Optional[int] = None,
) -> None:
    """Warm up for BART reconstruction."""
    # define dummy input for the warm up
    if num_coils:
        dummy_input = np.random.rand(1, num_readouts, num_spokes, num_coils)
    else:
        dummy_input = np.random.rand(1, num_readouts, num_spokes)
    # save dummy input
    writecfl(path_to_save_kspace, dummy_input)
    # reconstruct dummy input 10 times as a warm up
    for _ in range(
        10
    ):  # note: since the reconstruction time for the conventional approach is much longer, a reduced number of iterations for the warm up should be sufficient
        subprocess.run(cmp, shell=True, check=True)


def warm_up_nufft(
    method: str,
    num_spokes: int,
    num_readouts: int,
    path_traj_bart: Union[str, os.PathLike],
    path_to_save_kspace: Union[str, os.PathLike],
    path_to_recon_image: Union[str, os.PathLike],
    device: str,
) -> None:
    """Warm up for the evaluation of the NUFFT reconstructions."""
    cmp = return_command_for_nufft_reconstruction_bart(
        method,
        path_traj_bart,
        path_to_save_kspace,
        path_to_recon_image,
        device,
    )
    warm_up_bart(num_spokes, num_readouts, cmp, path_to_save_kspace, device)


def warm_up_compressed_sensing(
    num_spokes: int,
    num_readouts: int,
    path_traj_bart: Union[str, os.PathLike],
    path_to_save_kspace: Union[str, os.PathLike],
    path_to_recon_image: Union[str, os.PathLike],
    path_to_sensitivity_map: Union[str, os.PathLike],
    bart_maxiter: int,
    bart_regularization_option: str,
    bart_regularization: float,
    device: str,
    bart_bitmask: Optional[int] = None,
    stepsize: Optional[float] = None,
    num_coils: Optional[int] = None,
) -> None:
    """Warm up for the evaluation of the compressed sensing reconstructions."""
    cmp = return_command_for_compressed_sensing_reconstruction_bart(
        path_traj_bart,
        path_to_save_kspace,
        path_to_recon_image,
        path_to_sensitivity_map,
        bart_maxiter,
        bart_regularization_option,
        bart_regularization,
        device,
        bart_bitmask,
        stepsize,
    )

    warm_up_bart(num_spokes, num_readouts, cmp, path_to_save_kspace, device, num_coils)


def warm_up_bart_reconstruction(
    method: str,
    num_spokes: int,
    num_readouts: int,
    path_traj_bart: Union[str, os.PathLike],
    path_to_save_kspace: Union[str, os.PathLike],
    path_to_recon_image: Union[str, os.PathLike],
    device: str,
    path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
    bart_maxiter: Optional[int] = None,
    bart_regularization_option: Optional[str] = None,
    bart_regularization: Optional[float] = None,
    num_coils: Optional[int] = None,
    bart_bitmask: Optional[int] = None,
    stepsize: Optional[float] = None,
) -> None:
    """Warm up BART reconstruction."""
    if method == 'CS':
        if (
            path_to_sensitivity_map
            and bart_maxiter
            and bart_regularization_option
            and bart_regularization
        ):
            warm_up_compressed_sensing(
                num_spokes,
                num_readouts,
                path_traj_bart,
                path_to_save_kspace,
                path_to_recon_image,
                path_to_sensitivity_map,
                bart_maxiter,
                bart_regularization_option,
                bart_regularization,
                device,
                bart_bitmask,
                stepsize,
                num_coils,
            )
    elif method == 'nufft_adjoint' or method == 'nufft_inverse':
        warm_up_nufft(
            method,
            num_spokes,
            num_readouts,
            path_traj_bart,
            path_to_save_kspace,
            path_to_recon_image,
            device,
        )

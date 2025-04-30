"""Class and functions for reconstruction k-space measurements."""
import os
import tempfile
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from src.conventional_reconstruction.conventional_reconstruction_radial_measurements import (
    reconstruct_k_space_measurements_with_compressed_sensing,
    reconstruct_k_space_measurements_with_nufft_adjoint,
    reconstruct_k_space_measurements_with_nufft_inverse,
)
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    calculate_density_correction,
    return_command_for_compressed_sensing_reconstruction_bart,
    return_command_for_nufft_reconstruction_bart,
    save_sensitivity_map_bart,
    warm_up_compressed_sensing,
    warm_up_nufft,
)
from src.machine_learning.inference.ml_reconstruction_inference_mr_measurements import (
    reconstruct_multiple_coil_k_space_measurements_with_ml,
)
from src.machine_learning.inference.ml_reconstruction_inference_utils import warm_up_ml
from src.utils.save_and_load import (
    define_directory_to_save_evaluation,
    define_directory_to_save_reconstruction,
    define_save_name_results_mri_measurements,
    make_directory,
    save_in_devshm,
    writecfl,
)
from src.utils.twixtools.map_twix import map_twix


def load_radial_k_space_measurements(
    num_spokes: int,
    num_readouts: int,
    path_to_measurements: Union[str, os.PathLike],
    filename: str,
    selected_coils_list: Optional[List[int]],
    method: str,
    im_w: int,
) -> np.ndarray:
    """Load radial k-space data."""
    file_path_k_space = os.path.join(path_to_measurements, filename)
    mapped = map_twix(file_path_k_space)
    k_space_data = mapped[-1]['image']
    k_space_data = np.squeeze(k_space_data[:])

    if len(k_space_data.shape) == 5:
        k_space_data = k_space_data[:, :, :num_spokes, :, :]
        if selected_coils_list is not None:
            k_space_data = k_space_data[:, :, :, selected_coils_list, :]
    elif len(k_space_data.shape) == 4:
        k_space_data = k_space_data[:, :num_spokes, :, :]
        if selected_coils_list is not None:
            k_space_data = k_space_data[:, :, selected_coils_list, :]
    else:
        raise Exception('The k-space data has an unexpected shape.')
    return k_space_data


def reshape_k_space_data_and_save_for_bart_reconstruction(
    method: str,
    k_radial: np.ndarray,
    orientation: int,
    path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
    filename: str,
    image_number: int,
):
    """Reshape measured k-space data and save it in different format for the BART reconstruction."""
    filename_reshape = (
        filename[:-4]
        + '_orientation_'
        + str(orientation)
        + '_image_number_'
        + str(image_number)
    )
    if len(k_radial.shape) == 4:
        if orientation is None:
            raise Exception(f'An orientation must be defined.')
        else:
            k_radial = k_radial[orientation, :, :, :]
    elif len(k_radial.shape) == 3:
        k_radial = k_radial
    else:
        raise Exception(f'The k-space data has an unexpected shape {k_radial.shape}')
    k_radial = np.transpose(k_radial, (2, 0, 1))
    k_radial = np.expand_dims(k_radial, axis=0)
    writecfl(
        os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
        k_radial,
    )


def save_reconstruction(
    num_spokes: int,
    reconstruction: np.ndarray,
    method: str,
    device: str,
    orientation: int,
    path_to_save_results: Union[str, os.PathLike],
    image_number: int,
    filename: str,
    subfolder_name: Optional[int] = None,
) -> None:
    """Save evalution."""
    path_to_save_results_reconstruction = define_directory_to_save_reconstruction(
        path_to_save_results, num_spokes, method, subfolder_name
    )

    save_name = define_save_name_results_mri_measurements(
        filename, method, device, orientation, num_spokes, image_number
    )
    make_directory(path_to_save_results_reconstruction)

    reconstruction_path = os.path.join(path_to_save_results_reconstruction, save_name)
    np.save(
        reconstruction_path,
        reconstruction,
    )
    print(f'Save reconstruction to {reconstruction_path}.')


def save_reconstruction_time(
    num_spokes: int,
    reconstruction: np.ndarray,
    method: str,
    device: str,
    orientation: Optional[int],
    time_list: List[float],
    path_to_save_results: Union[str, os.PathLike],
    filename: str,
    subfolder_name: Optional[int] = None,
) -> None:
    """Save reconstruction time."""
    path_to_save_results_evaluation = define_directory_to_save_evaluation(
        path_to_save_results, num_spokes, method, subfolder_name
    )
    save_name = define_save_name_results_mri_measurements(
        filename, method, device, orientation, num_spokes, 'all'
    )
    make_directory(path_to_save_results_evaluation)

    reconstruction_time_path = os.path.join(
        path_to_save_results_evaluation, save_name + '_' + 'time'
    )
    np.save(
        reconstruction_time_path,
        time_list,
    )
    print(f'Save reconstruction time to {reconstruction_time_path}.')


def reshape_k_space_data_and_save_for_ml_reconstruction(
    k_radial: np.ndarray,
    num_spokes: int,
    num_readouts: int,
    num_coils: int,
    orientation: Optional[int],
    path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
    filename: str,
    image_number: int,
):
    """Reshape measured k-space data and save as a numpy array for the machine learning reconstruction."""
    filename_reshape = (
        filename[:-4]
        + '_orientation_'
        + str(orientation)
        + '_image_number_'
        + str(image_number)
    )
    if len(k_radial.shape) == 4:
        if orientation is None:
            raise Exception(f'An orientation must be defined.')
        else:
            k_radial_orientation = k_radial[orientation, :, :, :]
    elif len(k_radial.shape) == 3:
        k_radial_orientation = k_radial
    else:
        raise Exception(f'The k-space data has an unexpected shape {k_radial.shape}')
    k_radial_orientation = np.transpose(k_radial_orientation, axes=[1, 0, 2])
    k_radial_orientation = np.reshape(
        k_radial_orientation, (num_coils, num_spokes * num_readouts)
    )
    k_radial_orientation = np.stack(
        (np.real(k_radial_orientation), np.imag(k_radial_orientation)), axis=1
    )
    np.save(
        os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
        k_radial_orientation,
    )


def evaluate_ml_reconstruction_measurements(
    num_spokes: int,
    num_readouts: int,
    num_coils: int,
    model_fc: Any,
    device: str,
    path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
    filename: str,
    orientation: int,
    image_number: int,
    path_to_save_ML_model: Union[str, os.PathLike],
    timer,
    warm_up: bool,
    num_repeat: int,
    time_list: List[float],
) -> Tuple[np.ndarray, List[float]]:
    """Evaluate machine learning reconstruction of k-space measurements."""
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        # define path to k-space data
        filename_reshape = (
            filename[:-4]
            + '_orientation_'
            + str(orientation)
            + '_image_number_'
            + str(image_number)
            + '.npy'
        )

        path_to_save_mri_measurements_reshape_k_space_tmp = save_in_devshm(
            os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
            tmpdirname,
        )

        with torch.no_grad():
            # warm up
            if warm_up:
                warm_up_ml(
                    model_fc,
                    num_spokes,
                    num_readouts,
                    num_coils,
                    device,
                )

            # repeat the reconstruction num_repeat times to obtain a more precise time measurement
            for repeat in range(0, num_repeat):
                with timer:
                    k_radial = np.load(
                        path_to_save_mri_measurements_reshape_k_space_tmp
                    )
                    reconstruction = (
                        reconstruct_multiple_coil_k_space_measurements_with_ml(
                            model_fc, device, k_radial
                        )
                    )
                # append reconstruction time to list
                time_list.append(timer.execution_time)
    return reconstruction, time_list


def evaluate_nufft_reconstruction_measurements(
    method: str,
    num_spokes: int,
    num_readouts: int,
    traj_variant: str,
    traj_angle: int,
    traj_shift: int,
    traj_isotropy: int,
    im_w: int,
    num_coils: int,
    device: str,
    path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
    filename: str,
    orientation: int,
    image_number: int,
    path_traj_bart: Union[str, os.PathLike],
    timer,
    num_repeat: int,
    warm_up: bool,
    time_list: List[float],
) -> Tuple[np.ndarray, List[float]]:
    """Evaluate NUFFT reconstruction of k-space measurements."""
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        # define paths
        filename_reshape = (
            filename[:-4]
            + '_orientation_'
            + str(orientation)
            + '_image_number_'
            + str(image_number)
        )
        path_to_save_mri_measurements_reshape_k_space_tmp = save_in_devshm(
            os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
            tmpdirname,
        )
        path_to_reconstruction = tmpdirname + '/' + 'reconstructed_image_bart_tmp'

        if method == 'nufft_adjoint':
            dens_correction_nufft_adjoint = calculate_density_correction(
                num_spokes,
                num_readouts,
                traj_variant,
                traj_angle,
                traj_shift,
                traj_isotropy,
                im_w,
                num_coils,
            )
            path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr = (
                path_to_save_mri_measurements_reshape_k_space_tmp + '_dens_corr'
            )

        # save data in /dev/shm
        path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)

        # define BART command
        if method == 'nufft_adjoint':
            cmp = return_command_for_nufft_reconstruction_bart(
                method,
                path_traj_bart,
                path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr,
                path_to_reconstruction,
                device,
            )
        elif method == 'nufft_inverse':
            cmp = return_command_for_nufft_reconstruction_bart(
                method,
                path_traj_bart,
                path_to_save_mri_measurements_reshape_k_space_tmp,
                path_to_reconstruction,
                device,
            )

        # warm up
        if warm_up:
            warm_up_nufft(
                method,
                num_spokes,
                num_readouts,
                path_traj_bart,
                tmpdirname + '/' + 'radial_k_space_warmup',
                tmpdirname + '/' + 'reconstructed_image_warmup',
                device,
            )

        # repeat the reconstruction num_repeat times to obtain a more precise time measurement
        if method == 'nufft_adjoint':
            for repeat in range(0, num_repeat):
                with timer:
                    reconstruction = (
                        reconstruct_k_space_measurements_with_nufft_adjoint(
                            cmp,
                            path_to_reconstruction,
                            path_to_save_mri_measurements_reshape_k_space_tmp,
                            path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr,
                            dens_correction_nufft_adjoint,
                        )
                    )

                time_list.append(timer.execution_time)

        elif method == 'nufft_inverse':
            for repeat in range(0, num_repeat):
                with timer:
                    reconstruction = (
                        reconstruct_k_space_measurements_with_nufft_inverse(
                            cmp,
                            path_to_reconstruction,
                        )
                    )

                time_list.append(timer.execution_time)

    return reconstruction, time_list


def evaluate_compressed_sensing_reconstruction_measurements(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    num_coils: int,
    device: str,
    path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
    filename: str,
    orientation: int,
    image_number: int,
    path_traj_bart: Union[str, os.PathLike],
    timer,
    num_repeat: int,
    warm_up: bool,
    time_list: List[float],
    bart_regularization_option: Optional[str] = None,
    bart_maxiter: Optional[int] = None,
    bart_regularization: Optional[float] = None,
    bart_bitmask: Optional[int] = None,
    path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
    stepsize: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Evaluate compressed sensing  reconstruction of k-space measurements."""
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        # define paths
        filename_reshape = (
            filename[:-4]
            + '_orientation_'
            + str(orientation)
            + '_image_number_'
            + str(image_number)
        )
        path_to_reconstruction = tmpdirname + '/' + 'reconstructed_image_bart_tmp'

        path_to_espirit_calibration = os.path.join(tmpdirname, 'espirit_calibration')

        # save data in /dev/shm
        path_to_save_mri_measurements_reshape_k_space_tmp = save_in_devshm(
            os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
            tmpdirname,
        )
        path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)

        # define BART command
        cmp = return_command_for_compressed_sensing_reconstruction_bart(
            path_traj_bart,
            path_to_save_mri_measurements_reshape_k_space_tmp,
            path_to_reconstruction,
            path_to_espirit_calibration,
            bart_maxiter,
            bart_regularization_option,
            bart_regularization,
            device,
            bart_bitmask,
            stepsize,
        )

        # warm up
        if warm_up:
            path_to_sensitivity_map = save_in_devshm(
                path_to_sensitivity_map, tmpdirname
            )
            warm_up_compressed_sensing(
                num_spokes,
                num_readouts,
                path_traj_bart,
                tmpdirname + '/' + 'radial_k_space_warmup',
                tmpdirname + '/' + 'reconstructed_image_warmup',
                path_to_sensitivity_map,
                bart_maxiter,
                bart_regularization_option,
                bart_regularization,
                device,
                bart_bitmask,
                stepsize,
                num_coils,
            )

        # repeat the reconstruction num_repeat times to obtain a more precise time measurement
        for repeat in range(0, num_repeat):
            with timer:
                reconstruction = (
                    reconstruct_k_space_measurements_with_compressed_sensing(
                        cmp,
                        path_to_reconstruction,
                        path_to_save_mri_measurements_reshape_k_space_tmp,
                        im_w,
                        path_traj_bart,
                        tmpdirname,
                    )
                )

            # append reconstruction time to list
            time_list.append(timer.execution_time)

    return reconstruction, time_list

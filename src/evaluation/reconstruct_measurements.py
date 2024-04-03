"""Class and functions for reconstruction k-space measurements."""
import os
import tempfile
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from scipy.ndimage.interpolation import rotate
from src.conventional_reconstruction.conventional_reconstruction_radial_measurements import (
    reconstruct_k_space_measurements_with_bart,
)
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    define_trajectory_bart,
    return_command_for_bart_reconstruction,
    warm_up_bart_reconstruction,
)
from src.machine_learning.inference.ml_reconstruction_inference_mr_measurements import (
    reconstruct_multiple_coil_k_space_measurements_with_ml,
)
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
    load_ml_model,
    warm_up_ml,
)
from src.utils.save_and_load import (
    define_directory_to_save_evaluation,
    define_directory_to_save_reconstruction,
    define_save_name_results_mri_measurements,
    make_directory,
    save_in_devshm,
    writecfl,
)
from src.utils.trajectory import define_radial_trajectory


class ReconstructionMRMeasurements:
    """ReconstructionMRMeasurements Class."""

    def __init__(
        self,
        num_spokes: int,
        num_readouts: int,
        im_w: int,
        k_w: int,
        num_coils: int,
    ) -> None:
        """Initialize ReconstructionMRMeasurements."""
        self.num_spokes = num_spokes
        self.num_readouts = num_readouts
        self.im_w = im_w
        self.k_w = k_w
        self.num_coils = num_coils
        self.num_repeat = 100  # define how often the reconstruction should be repeated to achieve more precise time measurements

    def rotate_images(self, reconstruction, orientation):
        """Rotate images."""
        if orientation == 0:
            reconstruction = reconstruction
        if orientation == 1:
            reconstruction = rotate(reconstruction, -90)
        if orientation == 2:
            reconstruction = np.flip(reconstruction, axis=0)
        return reconstruction

    def save_reconstruction(
        self,
        reconstruction: np.ndarray,
        method: str,
        device: str,
        orientation: int,
        time_list: List[float],
        path_to_save_results: Union[str, os.PathLike],
        subfolder_name: Optional[int] = None,
    ) -> None:
        """Save evalution."""
        path_to_save_results_reconstruction = define_directory_to_save_reconstruction(
            path_to_save_results, self.num_spokes, method, subfolder_name
        )
        path_to_save_results_evaluation = define_directory_to_save_evaluation(
            path_to_save_results, self.num_spokes, method, subfolder_name
        )
        save_name = define_save_name_results_mri_measurements(
            method, device, orientation, self.num_spokes
        )
        make_directory(path_to_save_results_reconstruction)
        make_directory(path_to_save_results_evaluation)

        np.save(
            os.path.join(path_to_save_results_reconstruction, save_name),
            reconstruction,
        )

        np.save(
            os.path.join(path_to_save_results_evaluation, save_name + '_' + 'time'),
            time_list,
        )

    def reshape_k_space_data_and_save_for_ml_reconstruction(
        self,
        k_radial: np.ndarray,
        orientation: int,
        path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
        filename: str,
    ):
        """Reshape measured k-space data and save as a numpy array for the machine learning reconstruction."""
        filename_reshape = filename[:-4] + '_orientation_' + str(orientation)
        k_radial_orientation = k_radial[orientation, :, :, :]
        k_radial_orientation = np.transpose(k_radial_orientation, axes=[1, 0, 2])
        k_radial_orientation = np.reshape(
            k_radial_orientation, (self.num_coils, self.num_spokes * self.num_readouts)
        )
        k_radial_orientation = np.stack(
            (np.real(k_radial_orientation), np.imag(k_radial_orientation)), axis=1
        )
        np.save(
            os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
            k_radial_orientation,
        )

    def reshape_k_space_data_and_save_for_bart_reconstruction(
        self,
        method: str,
        k_radial: np.ndarray,
        orientation: int,
        path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
        filename: str,
    ):
        """Reshape measured k-space data and save it in different format for the BART reconstruction."""
        filename_reshape = filename[:-4] + '_orientation_' + str(orientation)
        k_radial = k_radial[orientation, :, :, :]
        k_radial = np.transpose(k_radial, (2, 0, 1))
        k_radial = np.expand_dims(k_radial, axis=0)
        writecfl(
            os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
            k_radial,
        )

    def evaluate_ml_reconstruction_measurements(
        self,
        model_name: str,
        device: str,
        path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
        filename: str,
        orientation: int,
        path_to_save_ML_model: Union[str, os.PathLike],
        timer,
        warm_up: bool,
    ) -> Tuple[np.ndarray, List[float]]:
        """Evaluate machine learning reconstruction of k-space measurements."""
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
            # define empty list for reconstruction time measurements
            time_list = []  # type: List[float]

            # load ML model
            model_fc = load_ml_model(
                self.num_spokes,
                self.num_readouts,
                self.im_w,
                self.num_coils,
                model_name,
                True,
                device,
                path_to_save_ML_model,
            )

            # define path to k-space data
            filename_reshape = (
                filename[:-4] + '_orientation_' + str(orientation) + '.npy'
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
                        self.num_spokes,
                        self.num_readouts,
                        self.num_coils,
                        device,
                    )

                # repeat the reconstruction num_repeat times to obtain a more precise time measurement
                for repeat in range(0, self.num_repeat):
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

    def evaluate_bart_reconstruction_measurements(
        self,
        method: str,
        device: str,
        path_to_save_mri_measurements_reshape: Union[str, os.PathLike],
        filename: str,
        orientation: int,
        path_traj_bart: Union[str, os.PathLike],
        timer,
        warm_up: bool,
        bart_regularization_option: Optional[str] = None,
        bart_maxiter: Optional[int] = None,
        bart_regularization: Optional[float] = None,
        bart_bitmask: Optional[int] = None,
        path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
        stepsize: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """Evaluate compressed sensing or NUFFT reconstruction of k-space measurements."""
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
            time_list = []  # type: List[float]

            if method == 'nufft_adjoint':
                bart_norm_factor_of_traj = np.pi / (self.num_readouts // 4)
                traj_bart_rescaled = (
                    define_trajectory_bart(self.num_spokes, self.num_readouts)
                    * bart_norm_factor_of_traj
                )

                os_radread = 2
                dens0 = (
                    0.98
                    / np.sqrt(np.pi * self.im_w * self.im_w)
                    * (2 / os_radread) ** 1.1
                )
                dens_corr_fn = (
                    lambda tra: (dens0**4 + np.sum(tra**2, axis=0) ** 2) ** 0.25
                )
                dens_correction_nufft_adjoint = dens_corr_fn(traj_bart_rescaled)
                dens_correction_nufft_adjoint = np.repeat(
                    dens_correction_nufft_adjoint[:, :, np.newaxis],
                    self.num_coils,
                    axis=-1,
                )
            else:
                dens_correction_nufft_adjoint = None

            # save data in /dev/shm
            filename_reshape = filename[:-4] + '_orientation_' + str(orientation)

            path_to_save_mri_measurements_reshape_k_space_tmp = save_in_devshm(
                os.path.join(path_to_save_mri_measurements_reshape, filename_reshape),
                tmpdirname,
            )

            if method == 'CS':
                path_to_sensitivity_map = save_in_devshm(
                    path_to_sensitivity_map, tmpdirname
                )

            path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)
            path_to_reconstruction = tmpdirname + '/' + 'reconstructed_image_bart_tmp'

            if method == 'nufft_adjoint':
                path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr = (
                    path_to_save_mri_measurements_reshape_k_space_tmp + '_dens_corr'
                )
                cmp = return_command_for_bart_reconstruction(
                    method,
                    path_traj_bart,
                    path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr,
                    path_to_reconstruction,
                    device,
                    path_to_sensitivity_map,
                    bart_maxiter,
                    bart_regularization_option,
                    bart_regularization,
                    bart_bitmask,
                    stepsize,
                )
            else:
                path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr = None
                cmp = return_command_for_bart_reconstruction(
                    method,
                    path_traj_bart,
                    path_to_save_mri_measurements_reshape_k_space_tmp,
                    path_to_reconstruction,
                    device,
                    path_to_sensitivity_map,
                    bart_maxiter,
                    bart_regularization_option,
                    bart_regularization,
                    bart_bitmask,
                    stepsize,
                )

            # warm up
            if warm_up:
                warm_up_bart_reconstruction(
                    method,
                    self.num_spokes,
                    self.num_readouts,
                    path_traj_bart,
                    tmpdirname + '/' + 'radial_k_space_warmup',
                    tmpdirname + '/' + 'reconstructed_image_warmup',
                    device,
                    path_to_sensitivity_map,
                    bart_maxiter,
                    bart_regularization_option,
                    bart_regularization,
                    self.num_coils,
                    bart_bitmask,
                    stepsize,
                )

            # repeat the reconstruction num_repeat times to obtain a more precise time measurement
            for repeat in range(0, self.num_repeat):
                with timer:
                    reconstruction = reconstruct_k_space_measurements_with_bart(
                        cmp,
                        method,
                        path_to_reconstruction,
                        path_to_save_mri_measurements_reshape_k_space_tmp,
                        path_to_save_mri_measurements_reshape_k_space_tmp_dens_corr,
                        dens_correction_nufft_adjoint,
                    )
                # append reconstruction time to list
                time_list.append(timer.execution_time)

        return reconstruction, time_list

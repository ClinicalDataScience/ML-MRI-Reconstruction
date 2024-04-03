"""Class and functions to evaluate the performance of reconstruction approaches for synthetic data."""
import os
import tempfile
import time
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from skimage.metrics import mean_squared_error, structural_similarity
from src.conventional_reconstruction.conventional_reconstruction_synthetic_data import (
    reconstruct_synthetic_k_space_data_with_bart,
)
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    define_trajectory_bart,
    return_command_for_bart_reconstruction,
    warm_up_bart_reconstruction,
)
from src.machine_learning.inference.ml_reconstruction_inference_synthetic_data import (
    reconstruct_synthetic_k_space_data_with_ml,
)
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
    calculate_normalization_factor,
    load_ml_model,
    warm_up_ml,
)
from src.utils.save_and_load import (
    convert_pytorch_tensor_to_numpy_array,
    define_save_name_evaluation_synthetic_data,
    load_numpy_array_magnitude_output,
    make_directory,
    save_in_devshm,
)
from src.utils.trajectory import define_radial_trajectory
from torch import Tensor
from tqdm import tqdm


class EvaluationSyntheticData:
    """Evaluate reconstruction approaches."""

    def __init__(self, num_spokes: int, num_readouts: int, im_w: int) -> None:
        """Initialize EvaluationSyntheticData."""
        self.num_spokes = num_spokes
        self.num_readouts = num_readouts
        self.im_w = im_w

    @staticmethod
    def quantify_image_quality(
        reconstruction: np.ndarray,
        gt: np.ndarray,
        MSE_list: List[float],
        SSIM_list: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Calculate image quality metrics and add values to list."""
        MSE = mean_squared_error(gt, reconstruction)
        SSIM = structural_similarity(
            gt,
            reconstruction,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        MSE_list.append(MSE)
        SSIM_list.append(SSIM)
        return MSE_list, SSIM_list

    def save_evaluation(
        self,
        method: str,
        device: str,
        metric_name: str,
        metric_values_list: List[float],
        path_to_save_results_evaluation: Union[str, os.PathLike],
        bart_maxiter: Optional[int] = None,
    ) -> None:
        """Save evalution."""
        metric_values_list = np.array(metric_values_list)
        file_name_evaluation = define_save_name_evaluation_synthetic_data(
            method, metric_name, self.num_spokes, device
        )
        np.save(
            os.path.join(path_to_save_results_evaluation, file_name_evaluation),
            metric_values_list,
        )

    def evaluate_reconstruction(
        self,
        reconstruction: Tensor,
        output_path: Union[str, os.PathLike],
        MSE_list: List[float],
        SSIM_list: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Quantify image qulaity and append it to list."""
        gt = load_numpy_array_magnitude_output(output_path)

        MSE_list, SSIM_list = EvaluationSyntheticData.quantify_image_quality(
            reconstruction, gt, MSE_list, SSIM_list
        )
        return MSE_list, SSIM_list

    def return_list_of_commands_for_bart_reconstruction_with_new_path_in_devshm_and_copy_to_devshm(
        self,
        method: str,
        dataset_test,
        tmpdirname: str,
        path_traj_bart: Union[str, os.PathLike],
        path_to_save_results_reconstruction: Union[str, os.PathLike],
        device: str,
        path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
        bart_regularization_option: Optional[str] = None,
        bart_regularization: Optional[float] = None,
        bart_maxiter: Optional[int] = None,
        bart_bitmask: Optional[int] = None,
        stepsize: Optional[float] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """Return list of commands for bart reconstruction and save in /dev/shm."""
        cmp_list = (
            []
        )  # list of commands which are executed for the reconstruction with BART
        path_to_k_space_list = []  # list of paths where k-space is saved
        output_path_list = []  # list of paths where the ground truths is saved
        path_to_save_results_reconstruction_image_list = (
            []
        )  # list of paths where reconstructions are saved in /dev/shm
        path_to_save_results_reconstruction_image_numpy_list = (
            []
        )  # list where reconstructions in numpy format are saved
        path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)

        make_directory(os.path.join(tmpdirname, 'reconstruction'))
        make_directory(os.path.join(tmpdirname, 'k_space'))

        if method == 'CS':
            path_to_sensitivity_map = save_in_devshm(
                path_to_sensitivity_map, tmpdirname
            )
        for index in range(len(dataset_test)):
            input_path, output_path = dataset_test[index]

            path_to_save_results_reconstruction_image_numpy = os.path.join(
                path_to_save_results_reconstruction, os.path.split(input_path)[1]
            )
            path_to_save_results_reconstruction_image = os.path.join(
                tmpdirname, 'reconstruction', os.path.split(input_path)[1]
            )

            input_path = save_in_devshm(input_path, os.path.join(tmpdirname, 'k_space'))

            output_path_list.append(output_path)
            path_to_save_results_reconstruction_image_numpy_list.append(
                path_to_save_results_reconstruction_image_numpy
            )
            path_to_save_results_reconstruction_image_list.append(
                path_to_save_results_reconstruction_image
            )

            path_to_k_space_list.append(input_path)

            cmp = return_command_for_bart_reconstruction(
                method,
                path_traj_bart,
                input_path,
                path_to_save_results_reconstruction_image,
                device,
                path_to_sensitivity_map,
                bart_maxiter,
                bart_regularization_option,
                bart_regularization,
                bart_bitmask,
                stepsize,
            )

            cmp_list.append(cmp)
        return (
            cmp_list,
            path_to_k_space_list,
            output_path_list,
            path_to_save_results_reconstruction_image_list,
            path_to_save_results_reconstruction_image_numpy_list,
        )

    def evaluate_ml_model(
        self,
        device: str,
        dataset_test,
        normalization_factor: float,
        model_name: str,
        path_to_save_results_reconstruction: Union[str, os.PathLike],
        path_to_save_ML_model: Union[str, os.PathLike],
        timer,
        num_spokes_dropout: int,
        warm_up: bool,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Evaluate performance of machine learning model."""
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
            time_list = []  # type: List[float]
            MSE_list = []  # type: List[float]
            SSIM_list = []  # type: List[float]

            batch_size = 1
            model_fc = load_ml_model(
                self.num_spokes,
                self.num_readouts,
                self.im_w,
                batch_size,
                model_name,
                True,
                device,
                path_to_save_ML_model,
            )
            normalization_spoke_dropout = calculate_normalization_factor(
                self.num_spokes, num_spokes_dropout
            )

            with torch.no_grad():
                if warm_up:
                    warm_up_ml(
                        model_fc, self.num_spokes, self.num_readouts, batch_size, device
                    )

                for index in tqdm(range(len(dataset_test))):
                    input_path, output_path = dataset_test[index]

                    input_path = save_in_devshm(input_path, tmpdirname)

                    with timer:
                        reconstruction = reconstruct_synthetic_k_space_data_with_ml(
                            input_path,
                            normalization_factor,
                            normalization_spoke_dropout,
                            device,
                            model_fc,
                        )

                    time_list.append(timer.execution_time)

                    if device == 'cpu':
                        reconstruction = convert_pytorch_tensor_to_numpy_array(
                            reconstruction
                        )
                        MSE_list, SSIM_list = self.evaluate_reconstruction(
                            reconstruction, output_path, MSE_list, SSIM_list
                        )

                        np.save(
                            os.path.join(
                                path_to_save_results_reconstruction,
                                os.path.split(input_path)[1],
                            ),
                            reconstruction,
                        )

        return time_list, MSE_list, SSIM_list

    def evaluate_bart_reconstruction(
        self,
        method: str,
        device: str,
        dataset_test,
        path_to_save_results_reconstruction: Union[str, os.PathLike],
        path_traj_bart: Union[str, os.PathLike],
        timer,
        warm_up: bool,
        bart_regularization_option: Optional[str] = None,
        bart_maxiter: Optional[int] = None,
        bart_regularization: Optional[float] = None,
        bart_bitmask: Optional[int] = None,
        path_to_sensitivity_map: Optional[Union[str, os.PathLike]] = None,
        stepsize: Optional[float] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Evaluate compressed sensing or (adjoint) NUFFT reconstruction."""
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
            # define empty list
            time_list = []  # type: List[float]
            MSE_list = []  # type: List[float]
            SSIM_list = []  # type: List[float]

            if method == 'nufft_adjoint':
                bart_norm_factor_of_traj = np.pi / (self.num_readouts // 4)
                traj_bart_rescaled = (
                    define_trajectory_bart(self.num_spokes, self.num_readouts)
                    * bart_norm_factor_of_traj
                )

                # preparation for density correction for NUFFT adjoint
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
            else:
                dens_correction_nufft_adjoint = None
            (
                cmp_list,
                path_to_k_space_list,
                output_path_list,
                path_to_save_results_reconstruction_image_list,
                path_to_save_results_reconstruction_image_numpy_list,
            ) = self.return_list_of_commands_for_bart_reconstruction_with_new_path_in_devshm_and_copy_to_devshm(
                method,
                dataset_test,
                tmpdirname,
                path_traj_bart,
                path_to_save_results_reconstruction,
                device,
                path_to_sensitivity_map,
                bart_regularization_option,
                bart_regularization,
                bart_maxiter,
                bart_bitmask,
                stepsize,
            )

            if warm_up:
                warm_up_bart_reconstruction(
                    method,
                    self.num_spokes,
                    self.num_readouts,
                    path_traj_bart,
                    os.path.join(tmpdirname, 'radial_k_space_warmup'),
                    os.path.join(tmpdirname, 'reconstructed_image_warmup'),
                    device,
                    path_to_sensitivity_map,
                    bart_maxiter,
                    bart_regularization_option,
                    bart_regularization,
                    bart_bitmask,
                    stepsize,
                )

            for (
                cmp,
                path_to_save_results_reconstruction_image,
                path_to_save_results_reconstruction_image_numpy,
                path_to_k_space,
                output_path,
            ) in tqdm(
                zip(
                    cmp_list,
                    path_to_save_results_reconstruction_image_list,
                    path_to_save_results_reconstruction_image_numpy_list,
                    path_to_k_space_list,
                    output_path_list,
                )
            ):
                with timer:
                    reconstruction = reconstruct_synthetic_k_space_data_with_bart(
                        cmp,
                        method,
                        path_to_save_results_reconstruction_image,
                        self.num_spokes,
                        self.im_w,
                        path_to_k_space,
                        dens_correction_nufft_adjoint,
                    )

                time_list.append(timer.execution_time)

                if device == 'cpu':
                    MSE_list, SSIM_list = self.evaluate_reconstruction(
                        reconstruction, output_path, MSE_list, SSIM_list
                    )
                    np.save(
                        path_to_save_results_reconstruction_image_numpy, reconstruction
                    )
        return time_list, MSE_list, SSIM_list

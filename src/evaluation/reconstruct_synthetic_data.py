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
    calculate_density_correction,
    return_command_for_bart_reconstruction,
    return_command_for_nufft_reconstruction_bart,
    warm_up_compressed_sensing,
    warm_up_nufft,
)
from src.machine_learning.inference.ml_reconstruction_inference_synthetic_data import (
    reconstruct_synthetic_k_space_data_with_ml,
)
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
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
from torch import Tensor
from tqdm import tqdm


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
    method: str,
    num_spokes: int,
    device: str,
    metric_name: str,
    metric_values_list: List[float],
    path_to_save_results_evaluation: Union[str, os.PathLike],
    bart_maxiter: Optional[int] = None,
) -> None:
    """Save evalution."""
    metric_values_list = np.array(metric_values_list)
    file_name_evaluation = define_save_name_evaluation_synthetic_data(
        method, metric_name, num_spokes, device
    )
    np.save(
        os.path.join(path_to_save_results_evaluation, file_name_evaluation),
        metric_values_list,
    )


def evaluate_reconstruction(
    reconstruction: Tensor,
    output_path: Union[str, os.PathLike],
    MSE_list: List[float],
    SSIM_list: List[float],
) -> Tuple[List[float], List[float]]:
    """Quantify image qulaity and append it to list."""
    gt = load_numpy_array_magnitude_output(output_path)

    MSE_list, SSIM_list = quantify_image_quality(
        reconstruction, gt, MSE_list, SSIM_list
    )
    return MSE_list, SSIM_list


def evaluate_ml_model(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    device: str,
    dataset_test,
    model_name: str,
    path_to_save_results_reconstruction: Union[str, os.PathLike],
    path_to_save_ML_model: Union[str, os.PathLike],
    timer,
    warm_up: bool,
) -> Tuple[List[float], List[float], List[float]]:
    """Evaluate performance of machine learning model."""
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        time_list = []  # type: List[float]
        MSE_list = []  # type: List[float]
        SSIM_list = []  # type: List[float]

        batch_size = 1
        model_fc = load_ml_model(
            num_spokes,
            num_readouts,
            im_w,
            batch_size,
            model_name,
            True,
            device,
            path_to_save_ML_model,
        )
        normalization_factor = im_w**2

        with torch.no_grad():
            if warm_up:
                warm_up_ml(model_fc, num_spokes, num_readouts, batch_size, device)

            for index in tqdm(range(len(dataset_test))):
                input_path, output_path = dataset_test[index]

                input_path = save_in_devshm(input_path, tmpdirname)

                with timer:
                    reconstruction = reconstruct_synthetic_k_space_data_with_ml(
                        input_path, device, model_fc, normalization_factor
                    )

                time_list.append(timer.execution_time)

                if device == 'cpu':
                    reconstruction = convert_pytorch_tensor_to_numpy_array(
                        reconstruction
                    )
                    MSE_list, SSIM_list = evaluate_reconstruction(
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


def evaluate_nufft_reconstruction_synthetic_data(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    method: str,
    device: str,
    dataset_test,
    path_to_save_results_reconstruction: Union[str, os.PathLike],
    path_traj_bart: Union[str, os.PathLike],
    timer,
    warm_up: bool,
    traj_variant: str,
    traj_angle: int,
    traj_shift: int,
    traj_isotropy: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Evaluate compressed sensing or (adjoint) NUFFT reconstruction."""
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        # define empty list
        time_list = []  # type: List[float]
        MSE_list = []  # type: List[float]
        SSIM_list = []  # type: List[float]

        if method == 'nufft_adjoint':
            dens_correction_nufft_adjoint = calculate_density_correction(
                num_spokes,
                num_readouts,
                traj_variant,
                traj_angle,
                traj_shift,
                traj_isotropy,
                im_w,
                None,
            )
        else:
            dens_correction_nufft_adjoint = None

        path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)

        make_directory(os.path.join(tmpdirname, 'reconstruction'))
        make_directory(os.path.join(tmpdirname, 'k_space'))

        if warm_up:
            warm_up_nufft(
                method,
                num_spokes,
                num_readouts,
                path_traj_bart,
                os.path.join(tmpdirname, 'radial_k_space_warmup'),
                os.path.join(tmpdirname, 'reconstructed_image_warmup'),
                device,
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

            cmp = return_command_for_nufft_reconstruction_bart(
                method,
                path_traj_bart,
                input_path,
                path_to_save_results_reconstruction_image,
                device,
            )

            with timer:
                reconstruction = reconstruct_synthetic_k_space_data_with_bart(
                    cmp,
                    method,
                    path_to_save_results_reconstruction_image,
                    num_spokes,
                    im_w,
                    input_path,
                    dens_correction_nufft_adjoint,
                )

            time_list.append(timer.execution_time)

            if device == 'cpu':
                MSE_list, SSIM_list = evaluate_reconstruction(
                    reconstruction, output_path, MSE_list, SSIM_list
                )
                np.save(path_to_save_results_reconstruction_image_numpy, reconstruction)
    return time_list, MSE_list, SSIM_list


def evaluate_compressed_sensing_reconstruction_synthetic_data(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
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
    """Evaluate compressed sensing reconstruction."""
    method = 'CS'
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmpdirname:
        # define empty list
        time_list = []  # type: List[float]
        MSE_list = []  # type: List[float]
        SSIM_list = []  # type: List[float]

        path_traj_bart = save_in_devshm(path_traj_bart, tmpdirname)

        make_directory(os.path.join(tmpdirname, 'reconstruction'))
        make_directory(os.path.join(tmpdirname, 'k_space'))

        path_to_sensitivity_map = save_in_devshm(path_to_sensitivity_map, tmpdirname)

        if warm_up:
            warm_up_compressed_sensing(
                num_spokes,
                num_readouts,
                path_traj_bart,
                os.path.join(tmpdirname, 'radial_k_space_warmup'),
                os.path.join(tmpdirname, 'reconstructed_image_warmup'),
                path_to_sensitivity_map,
                bart_maxiter,
                bart_regularization_option,
                bart_regularization,
                device,
                bart_bitmask,
                stepsize,
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

            with timer:
                reconstruction = reconstruct_synthetic_k_space_data_with_bart(
                    cmp,
                    method,
                    path_to_save_results_reconstruction_image,
                    num_spokes,
                    im_w,
                )

            time_list.append(timer.execution_time)

            if device == 'cpu':
                MSE_list, SSIM_list = evaluate_reconstruction(
                    reconstruction, output_path, MSE_list, SSIM_list
                )
                np.save(path_to_save_results_reconstruction_image_numpy, reconstruction)
    return time_list, MSE_list, SSIM_list

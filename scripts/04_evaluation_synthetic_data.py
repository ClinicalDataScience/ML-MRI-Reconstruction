"""Evaluate reconstruction approaches on synthetic data."""
import argparse
import logging
import os
import sys

import torch
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    save_radial_trajectory_bart,
)
from src.evaluation.reconstruct_synthetic_data import (
    evaluate_compressed_sensing_reconstruction_synthetic_data,
    evaluate_ml_model,
    evaluate_nufft_reconstruction_synthetic_data,
    save_evaluation,
)
from src.utils import set_seed
from src.utils.dataset import define_samples_in_dataset, make_dataset
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import (
    define_directory_to_save_evaluation,
    define_directory_to_save_reconstruction,
    define_ML_model_folder_name,
    define_path_to_radial_trajectory,
    define_path_to_sensitivity_map,
    load_config,
    make_directory,
)
from src.utils.time_measurements import select_timer


def main(
    num_spokes: int,
    method: str,
    device: str,
    num_readouts: int,
    im_w: int,
    k_w: int,
    traj_variant: str,
    traj_angle: int,
    traj_shift: int,
    traj_isotropy: int,
    model_name: str,
    bart_regularization_option: str,
    bart_maxiter: int,
    bart_regularization: float,
    bart_bitmask: int,
    stepsize: float,
    path_to_data: str,
    path_to_save: str,
    subfolder_name: str,
    folder_name_ml_model: str,
    path_to_split_csv: str,
    warm_up: bool,
    seed: int,
) -> None:
    """Evaluate reconstruction approach on synthetic data."""
    torch.cuda.empty_cache()

    set_seed.seed_all(seed)

    path_to_save_results = path_to_save + 'synthetic_data/'
    if method == 'ML':
        path_to_save_ML_model = define_ML_model_folder_name(
            path_to_save, 'ML_model', folder_name_ml_model, num_spokes
        )

    else:
        path_to_sensitivity_map = define_path_to_sensitivity_map(path_to_data)

        path_to_radial_trajectory_bart = define_path_to_radial_trajectory(
            path_to_data, num_spokes
        )
        save_radial_trajectory_bart(
            num_spokes,
            num_readouts,
            traj_variant,
            traj_angle,
            traj_shift,
            traj_isotropy,
            path_to_radial_trajectory_bart,
        )

    filelist_test = define_samples_in_dataset(path_to_split_csv, 'test')
    dataset_test = make_dataset(
        path_to_data, method, num_spokes, filelist_test, subfolder_name
    )

    timer = select_timer(device)

    path_to_save_results_reconstruction = define_directory_to_save_reconstruction(
        path_to_save=path_to_save_results,
        num_spokes=num_spokes,
        method=method,
        subfolder_name=subfolder_name,
    )

    path_to_save_results_evaluation = define_directory_to_save_evaluation(
        path_to_save=path_to_save_results,
        num_spokes=num_spokes,
        method=method,
        subfolder_name=subfolder_name,
    )
    make_directory(path_to_save_results_reconstruction)
    make_directory(path_to_save_results_evaluation)

    if method == 'ML':
        time_list, MSE_list, SSIM_list = evaluate_ml_model(
            num_spokes,
            num_readouts,
            im_w,
            device,
            dataset_test,
            model_name,
            path_to_save_results_reconstruction,
            path_to_save_ML_model,
            timer,
            warm_up,
        )
    elif method == 'nufft_adjoint' or method == 'nufft_inverse':
        (
            time_list,
            MSE_list,
            SSIM_list,
        ) = evaluate_nufft_reconstruction_synthetic_data(
            num_spokes,
            num_readouts,
            im_w,
            method,
            device,
            dataset_test,
            path_to_save_results_reconstruction,
            path_to_radial_trajectory_bart,
            timer,
            warm_up,
            traj_variant,
            traj_angle,
            traj_shift,
            traj_isotropy,
        )
    elif method == 'CS':
        (
            time_list,
            MSE_list,
            SSIM_list,
        ) = evaluate_compressed_sensing_reconstruction_synthetic_data(
            num_spokes,
            num_readouts,
            im_w,
            device,
            dataset_test,
            path_to_save_results_reconstruction,
            path_to_radial_trajectory_bart,
            timer,
            warm_up,
            bart_regularization_option,
            bart_maxiter,
            bart_regularization,
            bart_bitmask,
            path_to_sensitivity_map,
            stepsize,
        )

    save_evaluation(
        method, num_spokes, device, 'time', time_list, path_to_save_results_evaluation
    )

    if device == 'cpu':
        save_evaluation(
            method, num_spokes, device, 'MSE', MSE_list, path_to_save_results_evaluation
        )
        save_evaluation(
            method,
            num_spokes,
            device,
            'SSIM',
            SSIM_list,
            path_to_save_results_evaluation,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file_name', required=True, type=str, help='Name of config file'
    )
    parser.add_argument(
        '--num_spokes',
        required=True,
        type=int,
        help='Number of spokes in radial k-space',
    )

    parser.add_argument(
        '--traj_variant',
        type=str,
        choices=['radial'],
        required=False,
        default='radial',
        help='trajectory variant',
    )

    parser.add_argument(
        '--traj_angle',
        type=float,
        default=180,
        required=False,
        help='trajectory: maximum rotation angle of spokes',
    )

    parser.add_argument(
        '--traj_shift',
        type=float,
        default=0.0,
        required=False,
        help='trajectory: small rotation by fraction of spoke angle',
    )

    parser.add_argument(
        '--traj_isotropy',
        type=float,
        default=1.0,
        required=False,
        help='trajectory: isotropic: p=1.0, else phi = atan(p * tan(phi0))',
    )

    parser.add_argument(
        '--method',
        required=True,
        type=str,
        choices=['ML', 'CS', 'nufft_adjoint', 'nufft_inverse'],
        help='Name of method that should be evaluated',
    )
    parser.add_argument(
        '--device',
        required=True,
        type=str,
        choices=['cpu', 'cuda'],
        help='Device that is being used',
    )

    parser.add_argument(
        '--bart_regularization_option',
        required=False,
        default=None,
        choices=['l1', 'W', 'N', 'H', 'F', 'T', 'L', 'Q', 'I'],
        type=str,
        help='BART regularization option for compressed sensing reconstruction',
    )

    parser.add_argument(
        '--bart_regularization',
        required=False,
        default=None,
        type=float,
        help='BART regularization for compressed sensing reconstruction',
    )

    parser.add_argument(
        '--maxiter',
        required=False,
        default=None,
        type=int,
        help='Maximum number of iterations for compressed sensing reconstruction',
    )

    parser.add_argument(
        '--bart_bitmask',
        required=False,
        default=None,
        type=int,
        help='BART bitmask for compressed sensing reconstruction',
    )

    parser.add_argument(
        '--bart_stepsize',
        required=False,
        default=None,
        type=float,
        help='BART stepsize for compressed sensing reconstruction',
    )

    parser.add_argument(
        '--model_name',
        required=False,
        default='LinearFCNetwork',
        type=str,
        help='Name of model for machine learning reconstruction',
    )

    parser.add_argument(
        '--folder_name_ml_model',
        required=False,
        default=None,
        type=str,
        help='Name of folder where ML model was saved',
    )

    parser.add_argument(
        '--subfolder_name',
        required=False,
        default=None,
        type=str,
        help='Name of folder where the results should be saved and where the k-space data was saved',
    )

    parser.add_argument('--warm_up', action='store_true')

    # parse arguments
    args = parser.parse_args()

    if args.method == 'CS':
        if (
            args.maxiter is None
            or args.bart_regularization_option is None
            or args.bart_regularization is None
        ):
            parser.error(
                'If compressed sensing is selected as a reconstruction approach, a number of iterations, regularization option and strength of regularization must be defined.'
            )
        else:
            if args.bart_regularization_option != 'l1' and args.bart_bitmask is None:
                parser.error(
                    'If compressed sensing is selected as a reconstruction approach, a BART bitmask must be defined.'
                )

    config = load_config('configs/' + args.config_file_name)

    set_up_logging_and_save_args_and_config(
        'evaluation_synthetic_data_' + str(args.num_spokes) + '_spokes_' + args.method,
        args,
        config,
    )

    main(
        num_spokes=args.num_spokes,
        method=args.method,
        device=args.device,
        num_readouts=config['num_readouts'],
        im_w=config['im_w'],
        k_w=config['k_w'],
        traj_variant=args.traj_variant,
        traj_angle=args.traj_angle,
        traj_shift=args.traj_shift,
        traj_isotropy=args.traj_isotropy,
        model_name=args.model_name,
        bart_regularization_option=args.bart_regularization_option,
        bart_maxiter=args.maxiter,
        bart_regularization=args.bart_regularization,
        bart_bitmask=args.bart_bitmask,
        stepsize=args.bart_stepsize,
        path_to_data=config['path_to_data'],
        path_to_save=config['path_to_save'],
        subfolder_name=args.subfolder_name,
        folder_name_ml_model=args.folder_name_ml_model,
        path_to_split_csv=config['path_to_split_csv'],
        warm_up=args.warm_up,
        seed=config['seed'],
    )

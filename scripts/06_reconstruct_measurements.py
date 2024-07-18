"""Evaluate reconstruction approaches on k-space measurements."""
import argparse
import os
from pathlib import Path
from typing import List, Union

import torch
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    save_sensitivity_map_bart,
    save_sensitivity_map_bart_coils,
)
from src.evaluation.reconstruct_measurements import ReconstructionMRMeasurements
from src.machine_learning.dataset.custom_dataset import find_normalization_factor
from src.utils import set_seed
from src.utils.load_measurements import load_phantom_data_radial
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import (
    define_folder_name,
    define_ML_model_folder_name,
    define_path_to_radial_trajectory,
    load_config,
    make_directory,
)
from src.utils.time_measurements import select_timer


def main(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    k_w: int,
    model_name: str,
    subfolder_name: str,
    bart_regularization_option: str,
    bart_maxiter: int,
    bart_regularization: float,
    bart_bitmask: int,
    stepsize: float,
    path_to_measurements: Union[str, os.PathLike],
    path_to_data: Union[str, os.PathLike],
    path_to_save: Union[str, os.PathLike],
    path_to_normalization_factor_csv: Union[str, os.PathLike],
    method: str,
    folder_name_ml_model,
    device: str,
    filename: str,
    image_number: int,
    orientation: int,
    selected_coils_list: List[int],
    warm_up: bool,
    num_repeat: int,
    seed: int,
):
    """Evaluate reconstruction approach on k-space measurements."""
    torch.cuda.empty_cache()

    set_seed.seed_all(seed)

    num_coils = len(selected_coils_list)
    path_to_save_mri_measurements_reshape = os.path.join(
        path_to_save,
        'mri_measurements',
        'mri_measurements_preprocessed',
        method,
        'radial_k_' + str(num_spokes) + '_spokes',
    )
    make_directory(path_to_save_mri_measurements_reshape)

    path_to_save_results = os.path.join(path_to_save, 'mri_measurements')

    if method == 'ML':
        path_to_save_ML_model = define_ML_model_folder_name(
            path_to_save, 'ML_model', folder_name_ml_model, num_spokes
        )
    else:
        path_traj_bart = define_path_to_radial_trajectory(path_to_data, num_spokes)

        # Save a constant sensitivity map for warm up of compressed sensing
        if method == 'CS':
            if warm_up:
                path_to_sensitivity_map_folder = define_folder_name(
                    path_to_save, 'mri_measurements', 'sensitivity_map'
                )
                make_directory(path_to_sensitivity_map_folder)

                if num_coils == 1:
                    path_to_sensitivity_map = os.path.join(
                        path_to_sensitivity_map_folder, 'sensitivity_map_bart'
                    )
                    save_sensitivity_map_bart(im_w, path_to_sensitivity_map)

                else:
                    path_to_sensitivity_map = os.path.join(
                        path_to_sensitivity_map_folder,
                        'sensitivity_map_bart' + '_' + str(num_coils) + '_coils',
                    )

                    save_sensitivity_map_bart_coils(
                        im_w, num_coils, path_to_sensitivity_map
                    )
            else:
                path_to_sensitivity_map = None

    timer = select_timer(device)

    normalization_factor = find_normalization_factor(
        path_to_normalization_factor_csv, num_spokes
    )

    k_radial = load_phantom_data_radial(
        num_spokes,
        num_readouts,
        path_to_measurements,
        filename,
        image_number,
        selected_coils_list,
        method,
        im_w,
        normalization_factor,
    )

    reconstruction_mr_measurements = ReconstructionMRMeasurements(
        num_spokes, num_readouts, im_w, k_w, num_coils, num_repeat
    )

    if method == 'ML':
        reconstruction_mr_measurements.reshape_k_space_data_and_save_for_ml_reconstruction(
            k_radial,
            orientation,
            path_to_save_mri_measurements_reshape,
            filename,
        )
    else:
        reconstruction_mr_measurements.reshape_k_space_data_and_save_for_bart_reconstruction(
            method,
            k_radial,
            orientation,
            path_to_save_mri_measurements_reshape,
            filename,
        )

    if method == 'ML':
        (
            reconstruction,
            time_list,
        ) = reconstruction_mr_measurements.evaluate_ml_reconstruction_measurements(
            model_name,
            device,
            path_to_save_mri_measurements_reshape,
            filename,
            orientation,
            path_to_save_ML_model,
            timer,
            warm_up,
        )
    elif method == 'CS':
        (
            reconstruction,
            time_list,
        ) = reconstruction_mr_measurements.evaluate_bart_reconstruction_measurements(
            method,
            device,
            path_to_save_mri_measurements_reshape,
            filename,
            orientation,
            path_traj_bart,
            timer,
            warm_up,
            bart_regularization_option,
            bart_maxiter,
            bart_regularization,
            bart_bitmask,
            path_to_sensitivity_map,
            stepsize,
        )
    elif method == 'nufft_adjoint':
        (
            reconstruction,
            time_list,
        ) = reconstruction_mr_measurements.evaluate_bart_reconstruction_measurements(
            method,
            device,
            path_to_save_mri_measurements_reshape,
            filename,
            orientation,
            path_traj_bart,
            timer,
            warm_up,
        )

    reconstruction = reconstruction_mr_measurements.rotate_images(
        reconstruction, orientation
    )

    reconstruction_mr_measurements.save_reconstruction(
        reconstruction,
        method,
        device,
        orientation,
        time_list,
        path_to_save_results,
        image_number,
        subfolder_name,
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
        '--method',
        required=True,
        type=str,
        choices=['ML', 'CS', 'nufft_adjoint'],
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
        '--filename',
        required=True,
        type=str,
        help='Filename of the k-space measurements',
    )
    parser.add_argument(
        '--image_number',
        required=False,
        default=5,
        type=int,
        help='Image number that should be reconstructed',
    )
    parser.add_argument(
        '--orientation',
        required=False,
        default=1,
        type=int,
        choices=[0, 1, 2],
        help='Orientation that should be reconstructed',
    )

    parser.add_argument(
        '--bart_regularization_option',
        required=False,
        default=None,
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
        help='Name of model',
    )

    parser.add_argument(
        '--subfolder_name',
        required=False,
        default=None,
        type=str,
        help='Subfolder name for saving results',
    )

    parser.add_argument(
        '--folder_name_ml_model',
        required=False,
        default=None,
        type=str,
        help='Name of folder where ML model was saved',
    )

    parser.add_argument(
        '--selected_coils_list',
        required=True,
        help='List of coils for the reconstruction',
    )

    parser.add_argument(
        '--num_repeat',
        required=True,
        type=int,
        default=1,
        help='How often reconstruction is repeated',
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

    config = load_config('configs/' + args.config_file_name)
    set_up_logging_and_save_args_and_config(
        'reconstruct_measurements_' + str(args.num_spokes) + '_spokes_' + args.method,
        args,
        config,
    )

    selected_coils_list = args.selected_coils_list.split(',')
    selected_coils_list_int = [int(i) for i in selected_coils_list]

    main(
        num_spokes=args.num_spokes,
        num_readouts=config['num_readouts'],
        im_w=config['im_w'],
        k_w=config['k_w'],
        model_name=args.model_name,
        subfolder_name=args.subfolder_name,
        bart_regularization_option=args.bart_regularization_option,
        bart_maxiter=args.maxiter,
        bart_regularization=args.bart_regularization,
        bart_bitmask=args.bart_bitmask,
        stepsize=args.bart_stepsize,
        path_to_measurements=Path(config['path_to_measurements']),
        path_to_data=Path(config['path_to_data']),
        path_to_save=Path(config['path_to_save']),
        path_to_normalization_factor_csv=config['path_to_normalization_factor_csv'],
        method=args.method,
        folder_name_ml_model=args.folder_name_ml_model,
        device=args.device,
        filename=args.filename,
        image_number=args.image_number,
        orientation=args.orientation,
        selected_coils_list=selected_coils_list_int,
        warm_up=args.warm_up,
        num_repeat=args.num_repeat,
        seed=config['seed'],
    )

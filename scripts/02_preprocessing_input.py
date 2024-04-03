"""Generate radial k-space data from complex-valued synthetic images."""
import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
from src.conventional_reconstruction.conventional_reconstruction_synthetic_data import (
    reshape_input_for_bart_reconstruction,
)
from src.conventional_reconstruction.conventional_reconstruction_utils import (
    save_radial_trajectory_bart,
    save_sensitivity_map_bart,
)
from src.preprocessing.generate_radial_k_space import build_pynufft_object
from src.utils import set_seed
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import (
    define_path_to_radial_trajectory,
    define_path_to_sensitivity_map,
    define_radial_k_space_folder_name,
    list_files_in_directory,
    load_config,
    make_directory,
    save_real_and_imaginary_parts_in_2_channels,
    writecfl,
)
from src.utils.trajectory import define_radial_trajectory
from tqdm import tqdm


def main(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    k_w: int,
    interpolation_w: int,
    path_to_data: Union[str, os.PathLike],
    seed: int,
    noise: bool,
    subfolder_name: str,
) -> None:
    """Generate radial k-space data from complex-valued synthetic data."""
    set_seed.seed_all(seed)

    # define directories
    path_to_ground_truth = os.path.join(path_to_data, 'complex_image')
    path_to_input_ml = define_radial_k_space_folder_name(
        path_to_data, 'ML', subfolder_name, num_spokes
    )
    path_to_input_bart = define_radial_k_space_folder_name(
        path_to_data, 'BART', subfolder_name, num_spokes
    )
    make_directory(path_to_input_ml)
    make_directory(path_to_input_bart)

    # define sensitivity map and trajectory for BART
    path_to_sensitivity_map = define_path_to_sensitivity_map(path_to_data)
    if not Path(path_to_sensitivity_map).exists():
        save_sensitivity_map_bart(im_w, path_to_sensitivity_map)
    path_to_radial_trajectory_bart = define_path_to_radial_trajectory(
        path_to_data, num_spokes
    )
    save_radial_trajectory_bart(
        num_spokes, num_readouts, path_to_radial_trajectory_bart
    )

    NufftObj = build_pynufft_object(
        num_spokes,
        num_readouts,
        im_w,
        k_w,
        interpolation_w,
    )

    if noise:
        std_noise = 128 * np.sqrt(2) * 0.02

    filelist = list_files_in_directory(path_to_ground_truth, 'npy')

    for filename in tqdm(filelist):
        data = np.load(os.path.join(path_to_ground_truth, filename))

        data = data[0, :] + 1j * data[1, :]

        k_space = NufftObj.forward(data)

        if noise:
            k_space = (
                k_space
                + np.random.normal(0, std_noise, k_space.shape)
                + 1j * np.random.normal(0, std_noise, k_space.shape)
            )

        input_ml_model = save_real_and_imaginary_parts_in_2_channels(k_space)

        k_space_bart = reshape_input_for_bart_reconstruction(
            k_space, num_spokes, num_readouts, im_w
        )

        # save k-space
        file_path_input_ml = os.path.join(
            path_to_input_ml, os.path.splitext(filename)[0]
        )
        np.save(file_path_input_ml, input_ml_model)

        file_path_input_bart = os.path.join(
            path_to_input_bart, os.path.splitext(filename)[0]
        )
        writecfl(file_path_input_bart, k_space_bart)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file_name', metavar='path', required=True, help='Name of config file'
    )
    parser.add_argument(
        '--num_spokes',
        required=True,
        type=int,
        help='Number of spokes in radial k-space',
    )

    parser.add_argument(
        '--noise',
        action='store_true',
    )

    parser.add_argument(
        '--subfolder_name',
        required=False,
        default='radial_k',
        type=str,
        help='Name of folder where the k-space data should be saved',
    )

    args = parser.parse_args()

    config = load_config('configs/' + args.config_file_name)

    set_up_logging_and_save_args_and_config(
        'preprocessing_input_' + str(args.num_spokes) + '_spokes', args, config
    )

    if args.subfolder_name != 'noise' and args.noise is True:
        parser.error('The subfoldername must be called noise.')
    elif args.subfolder_name == 'noise' and args.noise is False:
        parser.error('The noise flag must be set.')

    main(
        num_spokes=args.num_spokes,
        num_readouts=config['num_readouts'],
        im_w=config['im_w'],
        k_w=config['k_w'],
        interpolation_w=config['interpolation_w'],
        path_to_data=Path(config['path_to_data']),
        seed=config['seed'],
        noise=args.noise,
        subfolder_name=args.subfolder_name,
    )

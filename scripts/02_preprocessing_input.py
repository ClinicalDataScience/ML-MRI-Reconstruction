"""Generate radial k-space data from complex-valued synthetic images."""
import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
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
    load_config,
    make_directory,
    save_real_and_imaginary_parts_in_2_channels,
    writecfl,
)
from src.utils.trajectory import define_trajectory
from tqdm import tqdm


def main(
    num_spokes: int,
    num_readouts: int,
    traj_variant: str,
    traj_angle: int,
    traj_shift: int,
    traj_isotropy: int,
    im_w: int,
    k_w: int,
    interpolation_w: int,
    path_to_data: Union[str, os.PathLike],
    path_to_split_csv: Union[str, os.PathLike],
    seed: int,
    noise: bool,
    subfolder_name: str,
    save_for_bart_reconstruction: bool,
    generate_k_space_for_train_set: bool,
    generate_k_space_for_validation_set: bool,
    generate_k_space_for_test_set: bool,
) -> None:
    """Generate radial k-space data from complex-valued synthetic data."""
    set_seed.seed_all(seed)

    # define directories
    path_to_ground_truth = os.path.join(path_to_data, 'complex_image')
    path_to_input_ml = define_radial_k_space_folder_name(
        path_to_data, 'ML', subfolder_name, num_spokes
    )
    make_directory(path_to_input_ml)

    if save_for_bart_reconstruction:
        path_to_input_bart = define_radial_k_space_folder_name(
            path_to_data, 'BART', subfolder_name, num_spokes
        )
        make_directory(path_to_input_bart)

        # define sensitivity map for BART reconstruction
        path_to_sensitivity_map = define_path_to_sensitivity_map(path_to_data)
        if not Path(path_to_sensitivity_map).exists():
            save_sensitivity_map_bart(im_w, path_to_sensitivity_map)

        # save radial trajectory for BART reconstruction
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

    traj = define_trajectory(
        num_spokes, num_readouts, traj_variant, traj_angle, traj_shift, traj_isotropy
    )

    NufftObj = build_pynufft_object(
        traj,
        im_w,
        k_w,
        interpolation_w,
    )

    filelist = []
    if generate_k_space_for_train_set:
        df_train = pd.read_csv(os.path.join(path_to_split_csv, 'train_samples.csv'))
        train_list = df_train['0'].tolist()
        filelist += train_list
    if generate_k_space_for_validation_set:
        df_validation = pd.read_csv(
            os.path.join(path_to_split_csv, 'validation_samples.csv')
        )
        validation_list = df_validation['0'].tolist()
        filelist += validation_list
    if generate_k_space_for_test_set:
        df_test = pd.read_csv(os.path.join(path_to_split_csv, 'test_samples.csv'))
        test_list = df_test['0'].tolist()
        filelist += test_list

    for filename in tqdm(filelist):
        data = np.load(os.path.join(path_to_ground_truth, filename))

        data = data[0, :] + 1j * data[1, :]

        k_space = NufftObj.forward(data)

        if noise:
            std_noise = im_w * np.sqrt(2) * 0.02
            k_space = (
                k_space
                + np.random.normal(0, std_noise, k_space.shape)
                + 1j * np.random.normal(0, std_noise, k_space.shape)
            )

        input_ml_model = save_real_and_imaginary_parts_in_2_channels(k_space)

        # save k-space
        file_path_input_ml = os.path.join(
            path_to_input_ml, os.path.splitext(filename)[0]
        )
        np.save(file_path_input_ml, input_ml_model)

        if save_for_bart_reconstruction:
            # only save reshaped test images for BART reconstruction
            if filename in test_list:
                k_space_bart = reshape_input_for_bart_reconstruction(
                    k_space, num_spokes, num_readouts, im_w
                )
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
        default=360,
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
        '--save_for_bart_reconstruction',
        action='store_true',
    )

    parser.add_argument(
        '--generate_k_space_for_train_set',
        action='store_true',
    )

    parser.add_argument(
        '--generate_k_space_for_validation_set',
        action='store_true',
    )

    parser.add_argument(
        '--generate_k_space_for_test_set',
        action='store_true',
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
        traj_variant=args.traj_variant,
        traj_angle=args.traj_angle,
        traj_shift=args.traj_shift,
        traj_isotropy=args.traj_isotropy,
        im_w=config['im_w'],
        k_w=config['k_w'],
        interpolation_w=config['interpolation_w'],
        path_to_data=Path(config['path_to_data']),
        path_to_split_csv=Path(config['path_to_split_csv']),
        seed=config['seed'],
        noise=args.noise,
        subfolder_name=args.subfolder_name,
        save_for_bart_reconstruction=args.save_for_bart_reconstruction,
        generate_k_space_for_train_set=args.generate_k_space_for_train_set,
        generate_k_space_for_validation_set=args.generate_k_space_for_validation_set,
        generate_k_space_for_test_set=args.generate_k_space_for_test_set,
    )

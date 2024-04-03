"""Calculate normalization factor for synthetic k-space data."""
import argparse
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from src.utils import set_seed
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import define_radial_k_space_folder_name, load_config
from tqdm import tqdm


def find_normalization_factor(
    filelist: List[Union[str, os.PathLike]], path_to_input_ml: Union[str, os.PathLike]
):
    """Find normalization factor by calculating maximum of maximum of absolute values in training dataset."""
    dataset_abs_max = []
    for filename in tqdm(filelist):
        X = np.load(os.path.join(path_to_input_ml, filename))
        item_abs_max = np.max(np.abs(X))
        dataset_abs_max.append(item_abs_max)
    normalization_factor = np.max(dataset_abs_max)
    return normalization_factor


def save_normalization_factor_df(
    num_spokes: int,
    normalization_factor: float,
    path_to_normalization_factor_csv: Union[str, os.PathLike],
) -> None:
    """Save dataframe with normalization factor or add a new line to an already existing csv file."""
    if not os.path.exists(path_to_normalization_factor_csv):
        os.makedirs(path_to_normalization_factor_csv)
    filename_normalization_factor_csv = os.path.join(
        path_to_normalization_factor_csv, 'normalization_factor.csv'
    )

    normalization_factor_spokes_dict = {
        'num_spokes': [num_spokes],
        'normalization_factor': [normalization_factor],
    }
    df = pd.DataFrame(normalization_factor_spokes_dict)

    if not os.path.isfile(filename_normalization_factor_csv):
        df.to_csv(filename_normalization_factor_csv, index=False, header=True)
    else:
        df_old = pd.read_csv(filename_normalization_factor_csv)
        if not any(df_old.num_spokes == num_spokes):
            df.to_csv(
                filename_normalization_factor_csv, mode='a', index=False, header=False
            )
        else:
            df_old.loc[
                df_old['num_spokes'] == num_spokes, 'normalization_factor'
            ] = normalization_factor
            df_old.to_csv(filename_normalization_factor_csv, index=False, header=True)


def main(
    num_spokes: int,
    path_to_data: Union[str, os.PathLike],
    path_to_normalization_factor_csv: Union[str, os.PathLike],
    path_to_split_csv: Union[str, os.PathLike],
    subfolder_name: str,
    seed: int,
) -> None:
    """Calculate normalization factor for synthetic k-space data."""
    set_seed.seed_all(seed)

    df = pd.read_csv(os.path.join(path_to_split_csv, 'train_samples.csv'))
    filelist = df['0'].tolist()

    path_to_input_ml = define_radial_k_space_folder_name(
        path_to_data, 'ML', subfolder_name, num_spokes
    )

    normalization_factor = find_normalization_factor(filelist, path_to_input_ml)

    save_normalization_factor_df(
        num_spokes, normalization_factor, path_to_normalization_factor_csv
    )


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
        '--subfolder_name',
        required=False,
        default='radial_k',
        type=str,
        help='Name of folder where the k-space data should be saved',
    )
    args = parser.parse_args()

    config = load_config('configs/' + args.config_file_name)

    set_up_logging_and_save_args_and_config(
        'preprocessing_input_normalization_' + str(args.num_spokes) + '_spokes',
        args,
        config,
    )

    main(
        num_spokes=args.num_spokes,
        path_to_data=Path(config['path_to_data']),
        path_to_normalization_factor_csv=Path(
            config['path_to_normalization_factor_csv']
        ),
        path_to_split_csv=Path(config['path_to_split_csv']),
        subfolder_name=args.subfolder_name,
        seed=config['seed'],
    )

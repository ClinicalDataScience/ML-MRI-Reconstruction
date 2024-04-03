"""Functions for defining a dataset."""
import os
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from src.utils.save_and_load import define_radial_k_space_folder_name


def define_samples_in_dataset(
    path_to_split_csv: Union[str, os.PathLike], status: str
) -> List[Union[str, os.PathLike]]:
    """Read the csv file where the sample names of the train, validation or test set are listed and convert it to a list."""
    df = pd.read_csv(os.path.join(path_to_split_csv, str(status) + '_samples.csv'))
    filelist = df['0'].tolist()
    return filelist


def make_dataset(
    path_to_data: Union[str, os.PathLike],
    method: str,
    num_spokes: int,
    filelist: List[Union[str, os.PathLike]],
    subfolder_name: Optional[str] = 'radial_k',
) -> Sequence[Tuple[Union[str, os.PathLike], Union[str, os.PathLike]]]:
    """Return a dataset as a list of tuples of paired image paths: (input_path, output_path)."""
    dataset = []

    if subfolder_name is None:
        subfolder_name = 'radial_k'

    # define paths to input and output
    if method == 'ML':
        path_to_input_ml = define_radial_k_space_folder_name(
            path_to_data, 'ML', subfolder_name, num_spokes
        )
    else:
        path_to_input_bart = define_radial_k_space_folder_name(
            path_to_data, 'BART', subfolder_name, num_spokes
        )

    # fill dataset list with tuples of paths to input and output
    for output_fname in filelist:
        if method == 'ML':
            input_path = os.path.join(path_to_input_ml, output_fname)
        else:
            input_path = os.path.join(
                path_to_input_bart, str(output_fname).rsplit('.', 1)[0]
            )

        output_path = os.path.join(path_to_data, 'complex_image', output_fname)
        item = (input_path, output_path)
        dataset.append(item)

    return dataset

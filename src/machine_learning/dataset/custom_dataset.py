"""Functions for defining a dataset for this project."""
import os
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from src.utils.dataset import make_dataset
from src.utils.save_and_load import (
    load_numpy_array_convert_to_tensor_input,
    load_numpy_array_convert_to_tensor_output,
)
from torch import Tensor
from torch.utils.data import Dataset


def find_normalization_factor(
    path_to_normalization_factor_csv: Union[str, os.PathLike], num_spokes: int
) -> float:
    """Read the csv file where the normalization factors are saved and identify the correct one."""
    # read csv file where normalization factors are saved
    df = pd.read_csv(
        os.path.join(path_to_normalization_factor_csv, 'normalization_factor.csv')
    )
    # idenfity correct normalization factor based on number of spokes
    normalization_factor = df.loc[df['num_spokes'] == num_spokes][
        'normalization_factor'
    ]
    normalization_factor = float(normalization_factor.values)
    return normalization_factor


class CustomDataset(Dataset):
    """Build a dataset of synthetic data for this project."""

    def __init__(
        self,
        path_to_data: Union[str, os.PathLike],
        num_spokes: int,
        filelist: List[Union[str, os.PathLike]],
        normalization_factor: float,
        subfolder_name: Optional[str] = 'radial_k',
    ) -> None:
        """Initialize CustomDataset."""
        super().__init__()
        self.normalization_factor = normalization_factor
        self.loader_input = load_numpy_array_convert_to_tensor_input
        self.loader_output = load_numpy_array_convert_to_tensor_output
        self.samples = make_dataset(
            path_to_data, 'ML', num_spokes, filelist, subfolder_name
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get dataset element."""
        # get paths to input and output files
        input_path, output_path = self.samples[index]

        # load each image using loader
        input_sample = self.loader_input(input_path, self.normalization_factor)
        output_sample = self.loader_output(output_path)
        return input_sample.float(), output_sample.float()

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.samples)

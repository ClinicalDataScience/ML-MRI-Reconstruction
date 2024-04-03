"""Functions for loading k-space measurements."""
import os
from typing import List, Union

import numpy as np
from src.utils.twixtools.map_twix import map_twix


def load_phantom_data_radial(
    num_spokes: int,
    num_readouts: int,
    path_to_measurements: Union[str, os.PathLike],
    filename: str,
    image_number: int,
    selected_coils_list: List[int],
    method: str,
    im_w: int,
    normalization_factor_synthetic_data: float,
) -> np.ndarray:
    """Load radial k-space data of phantom."""
    file_path_k_space = os.path.join(path_to_measurements, filename)
    # load data
    mapped = map_twix(file_path_k_space)
    k_space_data = mapped[-1]['image']
    # select data for defined image number
    k_space_data_image = np.squeeze(k_space_data[:])[image_number, :, :num_spokes, :, :]
    # select coils
    k_space_data_image = k_space_data_image[:, :, selected_coils_list, :]

    # normalize input
    k_space_data_image = k_space_data_image / max(
        np.max(np.abs(np.real(k_space_data_image))),
        np.max(np.abs(np.imag(k_space_data_image))),
    )
    if method == 'nufft_adjoint' or method == 'CS':
        k_space_data_image = (
            k_space_data_image * (normalization_factor_synthetic_data) / im_w
        )
    return k_space_data_image


def load_phantom_data_cartesian(
    path_to_measurements: Union[str, os.PathLike],
    filename: str,
    image_number: int,
    selected_coils_list: List[int],
) -> np.ndarray:
    """Load Cartesian k-space data of phantom."""
    file_path_k_space = os.path.join(path_to_measurements, filename)
    # load data
    mapped = map_twix(file_path_k_space)
    k_space_data = mapped[-1]['image']
    # select data for defined image number
    k_space_data_image = np.squeeze(k_space_data[:])[image_number, :, :, :, :]
    # select coils
    k_space_data_image = k_space_data_image[:, :, selected_coils_list, :]
    return k_space_data_image

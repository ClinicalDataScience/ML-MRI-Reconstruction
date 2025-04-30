"""Functions for defining directory and file names and saving and loading data."""
import glob
import json
import logging
import mmap
import os
import shutil
import sys
from pathlib import Path
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np
import torch
from torch import Tensor


def make_directory(new_directory: Union[str, os.PathLike]) -> None:
    """Create a directory."""
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)


def define_folder_name(
    path_to_data: Union[str, os.PathLike],
    reconstruction_approach: str,
    subfolder_name: str,
) -> Union[str, os.PathLike]:
    """Build a folder name based on path and two subfolder names."""
    path_to_folder = os.path.join(path_to_data, reconstruction_approach, subfolder_name)
    return path_to_folder


def add_subfolder_for_spokes(
    path_to_folder: Union[str, os.PathLike], num_spokes: int
) -> Union[str, os.PathLike]:
    """Define subfolder name for different undersampling factors."""
    subfolder_name = str(num_spokes) + '_spokes'
    path_to_folder = os.path.join(path_to_folder, subfolder_name)
    return path_to_folder


def define_radial_k_space_folder_name(
    base_path: Union[str, os.PathLike],
    reconstruction_approach: str,
    subfolder_name: str,
    num_spokes: int,
) -> Union[str, os.PathLike]:
    """Define name of folder where radial k-space data is saved."""
    if subfolder_name != 'radial_k':
        subfolder_name = 'radial_k_' + subfolder_name
    path_to_folder = define_folder_name(
        base_path, reconstruction_approach, subfolder_name
    )
    path_to_folder = add_subfolder_for_spokes(path_to_folder, num_spokes)
    return path_to_folder


def define_path_to_sensitivity_map(
    path_to_data: Union[str, os.PathLike]
) -> Union[str, os.PathLike]:
    """Define name of folder where the sensitivity map is saved."""
    path_to_sensitivity_map_bart = define_folder_name(
        path_to_data, 'BART', 'sensitivity_map'
    )
    make_directory(path_to_sensitivity_map_bart)
    path_to_sensitivity_map = os.path.join(
        path_to_sensitivity_map_bart, 'sensitivity_map_bart'
    )
    return path_to_sensitivity_map


def define_path_to_radial_trajectory(
    path_to_data: Union[str, os.PathLike], num_spokes: int
) -> Union[str, os.PathLike]:
    """Define name of folder where the radial trajectory for the bART reconstruction is saved."""
    path_to_radial_trajectory_bart = define_folder_name(
        path_to_data, 'BART', 'radial_trajectory'
    )
    make_directory(path_to_radial_trajectory_bart)
    path_traj_bart = os.path.join(
        path_to_radial_trajectory_bart,
        'radial_trajectory_bart_' + str(num_spokes) + '_spokes',
    )
    return path_traj_bart


def define_ML_model_folder_name(
    path_to_results: Union[str, os.PathLike],
    subfolder_name: str,
    folder_name_ml_model: str,
    num_spokes: int,
) -> Union[str, os.PathLike]:
    """Define name of folder where ML model is saved."""
    path_to_folder = os.path.join(path_to_results, subfolder_name)
    if folder_name_ml_model is None:
        path_to_subfolder_spokes = add_subfolder_for_spokes(path_to_folder, num_spokes)
    else:
        path_to_subfolder = os.path.join(path_to_folder, folder_name_ml_model)
        path_to_subfolder_spokes = add_subfolder_for_spokes(
            path_to_subfolder, num_spokes
        )
    return path_to_subfolder_spokes


def save_in_devshm(file_directory: str, tmpdirname: str) -> Union[str, os.PathLike]:
    """Save file in /dev/shm."""
    if file_directory.endswith('.npy'):
        head, tail = os.path.split(file_directory)
        path_to_save_tmp = os.path.join(tmpdirname, tail)
        shutil.copy2(file_directory, path_to_save_tmp)
    else:
        for file in glob.glob(file_directory + '.*'):
            if file.endswith('.hdr') or file.endswith('.cfl'):
                head, tail = os.path.split(file)
                path_to_save_tmp_file = os.path.join(tmpdirname, tail)
                shutil.copy2(file, path_to_save_tmp_file)
                path_to_save_tmp = os.path.splitext(path_to_save_tmp_file)[0]
            else:
                logging.warning('An error has occured.')
                sys.exit('An error has occured.')

    return path_to_save_tmp


def load_config(config_path: Union[str, os.PathLike]) -> dict:
    """Load config file."""
    with open(config_path) as json_config_file:
        config = json.load(json_config_file)
    return config


def writecfl(name: str, array: np.ndarray) -> None:
    """
    Write cfl file.

    Source: https://github.com/mrirecon/bart/blob/master/python/cfl.py

    Use of this source code is governed by a BSD-3-Clause license.

    Copyright (c) 2013-2018. The Regents of the University of California.
    Copyright (c) 2013-2024. BART Developer Team and Contributors.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    with open(name + '.hdr', 'wt') as h:
        h.write('# Dimensions\n')
        for i in array.shape:
            h.write('%d ' % i)
        h.write('\n')
    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize
    with open(name + '.cfl', 'a+b') as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()


def readcfl(name: str) -> np.ndarray:
    """
    Read cfl file.

    Source: https://github.com/mrirecon/bart/blob/master/python/cfl.py

    Use of this source code is governed by a BSD-3-Clause license.

    Copyright (c) 2013-2018. The Regents of the University of California.
    Copyright (c) 2013-2024. BART Developer Team and Contributors.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    # get dims from .hdr
    with open(name + '.hdr', 'rt') as h:
        h.readline()  # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    with open(name + '.cfl', 'rb') as d:
        a = np.fromfile(d, dtype=np.complex64, count=n)
    return a.reshape(dims, order='F')  # column-major


def save_real_and_imaginary_parts_in_2_channels(data: np.ndarray) -> np.ndarray:
    """Save real and imaginary part in two channels."""
    data = np.squeeze(data)
    if data.ndim == 1:
        output_data = np.empty([2, data.shape[0]])
    elif data.ndim == 2:
        output_data = np.empty(([2, data.shape[0], data.shape[1]]))
    else:
        logging.warning('This function is not written for such an input size.')
        sys.exit('This function is not written for such an input size.')
    output_data[0, :] = data.real
    output_data[1, :] = data.imag
    return output_data


def load_numpy_array_convert_to_tensor_input(
    path: Union[str, os.PathLike], normalization_factor: float
) -> Tensor:
    """Load npy file from path, normalize the input with the normalization factor and convert it to a pytorch tensor."""
    # load data
    X = np.load(path)
    # normalize
    X = X / normalization_factor
    # convert numpy array to pytorch tensor
    X = torch.from_numpy(X)
    return X


def load_numpy_array_convert_to_tensor_output(path: Union[str, os.PathLike]) -> Tensor:
    """Load numpy array and convert it to a pytorch tensor."""
    X = np.load(path)
    X = torch.from_numpy(X)
    return X


def load_numpy_array_input(input_path: Union[str, os.PathLike]) -> np.ndarray:
    """Load input k-space and reshape it to a complex-valued k-space."""
    X = np.load(input_path)
    X = X[0, :] + 1j * X[1, :]
    X = np.squeeze(X)
    return X


def load_numpy_array_magnitude_output(
    output_path: Union[str, os.PathLike]
) -> np.ndarray:
    """Load output image and calculate magnitude image."""
    X = np.load(output_path)
    X = X[0, :, :] + 1j * X[1, :, :]
    X = np.abs(X)
    X = np.squeeze(X)
    return X


def load_pytorch_tensor_input(
    input_path: Union[str, os.PathLike], normalization_factor: float
) -> Tensor:
    """Load input k-space as a PyTorch tensor."""
    # load and normalize input
    X = load_numpy_array_convert_to_tensor_input(input_path, normalization_factor)
    X = X.unsqueeze(0)
    return X.float()


def convert_pytorch_tensor_to_numpy_array(X: Tensor) -> np.ndarray:
    """Transform PyTorch tensor to a numpy array."""
    return np.squeeze(X.cpu().detach().numpy())


def list_files_in_directory(
    directory: str, data_type: str
) -> List[Union[str, os.PathLike]]:
    """Return list of filenames in directory."""
    filelist = [_ for _ in sorted(os.listdir(directory)) if _.endswith(data_type)]
    return filelist


def define_directory_to_save_evaluation(
    path_to_save: Union[str, os.PathLike],
    num_spokes: int,
    method: str,
    subfolder_name: Optional[str] = None,
):
    """Return name of directory where the evaluation is saved."""
    if subfolder_name is None:
        path_to_save_results_evaluation = os.path.join(
            path_to_save, 'evaluation', method
        )

    else:
        path_to_save_results_evaluation = os.path.join(
            path_to_save, 'evaluation', method, subfolder_name
        )
    path_to_save_results_evaluation_spokes = add_subfolder_for_spokes(
        path_to_save_results_evaluation, num_spokes
    )
    return path_to_save_results_evaluation_spokes


def define_directory_to_save_reconstruction(
    path_to_save: Union[str, os.PathLike],
    num_spokes: int,
    method: str,
    subfolder_name: Optional[str] = None,
):
    """Return name of directory where the reconstruction is saved."""
    if subfolder_name is None:
        path_to_save_results_reconstruction = os.path.join(
            path_to_save, 'reconstruction', method
        )

    else:
        path_to_save_results_reconstruction = os.path.join(
            path_to_save, 'reconstruction', method, subfolder_name
        )

    path_to_save_results_reconstruction_spokes = add_subfolder_for_spokes(
        path_to_save_results_reconstruction, num_spokes
    )
    return path_to_save_results_reconstruction_spokes


def define_save_name_evaluation_synthetic_data(
    method: str,
    metric_name: str,
    num_spokes: int,
    device: str,
) -> str:
    """Define the filename for the results of the evaluation of the reconstruction for synthetic data."""
    save_name = (
        method
        + '_'
        + 'device'
        + '_'
        + device
        + '_'
        + str(num_spokes)
        + '_'
        + 'spokes'
        + '_'
        + str(metric_name)
        + '_'
        + 'evaluation'
    )
    return save_name


def define_save_name_results_mri_measurements(
    filename: str,
    method: str,
    device: str,
    orientation: int,
    num_spokes: int,
    image_number: Union[int, str],
) -> str:
    """Define the filename for the results of the reconstruction for k-space measurements data."""
    if filename.endswith(('.dat')):
        filename = filename[:-4]
    save_name = (
        method
        + '_'
        + 'device'
        + '_'
        + device
        + '_'
        + str(filename)
        + '_'
        + 'orientation'
        + '_'
        + str(orientation)
        + '_'
        + 'imagenumber'
        + '_'
        + str(image_number)
        + '_'
        + 'spokes'
        + '_'
        + str(num_spokes)
    )
    return save_name

"""Functions for setting up a reconstruction with the machine learning model."""
import logging
import os
import sys
from typing import Union

import numpy as np
import torch
from src.machine_learning.model.model_fc import LinearFCNetwork


def load_ml_model(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    batch_size: int,
    model_name: str,
    calculate_magnitude: bool,
    device: str,
    path_to_save: Union[str, os.PathLike],
):
    """Load machine learning model."""
    # load inference model
    if model_name == 'LinearFCNetwork':
        model_fc = LinearFCNetwork(num_spokes, num_readouts, im_w, calculate_magnitude)
    else:
        logging.warning('This model is not defined.')
        sys.exit('This model is not defined.')

    model_cp = torch.load(
        os.path.join(
            path_to_save, 'ML_' + str(num_spokes) + '_spokes' + '_model_best.pth'
        ),
        map_location=torch.device(device),
    )
    model_fc.load_state_dict(model_cp['model_state_dict'])

    # send model to device
    model_fc.to(device)

    if device == 'cpu':
        # enable oneDNN Graph fusion
        torch.jit.enable_onednn_fusion(True)
    with torch.no_grad():
        # set model to evaluation mode
        model_fc.eval()
        if device == 'cpu':
            # create a random sample input
            sample_input = [
                torch.rand(batch_size, 2, num_spokes * num_readouts).float().to(device)
            ]
            # tracing the model with example input
            model_fc = torch.jit.trace(model_fc, sample_input)
            # invoking torch.jit.freeze
            model_fc = torch.jit.freeze(model_fc)
    return model_fc


def apply_ml_model_to_input(X: np.ndarray, model) -> np.ndarray:
    """Apply machine learning model to input."""
    reconstruction = model(X)
    return reconstruction


def warm_up_ml(
    model, num_spokes: int, num_readouts: int, batch_size: int, device: str
) -> None:
    """Warm up for the evaluation of the reconstructions with the machine learning model by reconstructing a random dummy input 100 times."""
    dummy_input = torch.rand(
        (batch_size, 2, num_spokes * num_readouts), dtype=torch.float
    ).to(device)
    for _ in range(100):
        _ = model(dummy_input)


def calculate_normalization_factor(num_spokes: int, num_spokes_dropout: int):
    """Calculate normalization factor for the input data during validation/testing for the reconstructions with the machine learning model."""
    normalization_spoke_dropout = (num_spokes - num_spokes_dropout) / num_spokes
    return normalization_spoke_dropout

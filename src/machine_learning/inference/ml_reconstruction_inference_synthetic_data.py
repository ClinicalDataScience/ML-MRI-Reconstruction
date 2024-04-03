"""Function for reconstruction synthetic data with a trained machine learning model."""
import os
from typing import Union

import numpy as np
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
    apply_ml_model_to_input,
)
from src.utils.save_and_load import load_pytorch_tensor_input


def reconstruct_synthetic_k_space_data_with_ml(
    input_path: Union[str, os.PathLike],
    normalization_factor: float,
    normalization_spoke_dropout: float,
    device: str,
    model,
) -> np.ndarray:
    """Reconstruct synthetic k-space data."""
    X = load_pytorch_tensor_input(input_path, normalization_factor)
    # normalize input when model was trained with spoke dropout (otherwise normalization_spoke_dropout is 1)
    X = X * normalization_spoke_dropout
    X = X.to(device)
    reconstruction = apply_ml_model_to_input(X, model)
    return reconstruction

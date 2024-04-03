"""Function for reconstruction k-space measurements with a trained ML model."""
import numpy as np
import torch
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
    apply_ml_model_to_input,
)
from src.utils.mathematical_operations import root_sum_of_squares
from src.utils.save_and_load import convert_pytorch_tensor_to_numpy_array


def reconstruct_k_space_measurements_with_ml(
    model, device: str, X: np.ndarray
) -> np.ndarray:
    """Reconstruct radial k-space data of one coil with machine learning."""
    X = torch.from_numpy(X)
    X = X.float()
    X = X.to(device)
    reconstruction = apply_ml_model_to_input(X, model)
    return reconstruction


def reconstruct_multiple_coil_k_space_measurements_with_ml(
    model, device: str, k_radial: np.ndarray
) -> np.ndarray:
    """Reconstruct radial k-space data of multiple coils with machine learning by combining the reconstruction of a single coil with a root-sum-of-squares algorithm."""
    reconstruction = reconstruct_k_space_measurements_with_ml(model, device, k_radial)
    reconstruction = convert_pytorch_tensor_to_numpy_array(reconstruction)
    reconstruction = root_sum_of_squares(reconstruction, 0)
    return reconstruction

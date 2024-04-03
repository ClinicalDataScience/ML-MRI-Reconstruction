"""Functions for generating synthetic complex-valued images."""
import logging
import os
import random
import sys

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from src.utils.normalization import min_max_normalization


def rgba2rgb(rgba: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
    """
    Convert RGBA image to RGB image.

    This function is from an answer (https://stackoverflow.com/a/58748986) by Feng Wang (https://stackoverflow.com/users/1475287/feng-wang) to the question "Convert RGBA to RGB in Python" (https://stackoverflow.com/q/50331463) on Stack Overflow, lincensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
    """
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype='uint8')


def tranform_to_greyscale(image: np.ndarray) -> np.ndarray:
    """Convert image to greyscale image."""
    # RGB image to greyscale image
    if image.ndim == 3 and (image.shape)[2] == 3:
        image_greyscale = rgb2gray(image)
    # RGB-A image to RGB image to greyscale image
    elif image.ndim == 3 and (image.shape)[2] == 4:
        image_rgb = rgba2rgb(image)
        image_greyscale = rgb2gray(image_rgb)
    # image is already greyscale
    elif image.ndim == 2:
        if image.dtype == 'uint8':
            image_greyscale = image / 255
        elif np.min(image) >= 0 and np.max(image) <= 1:
            image_greyscale = image
        else:
            logging.warning('The greyscale image has an unexpected format.')
            sys.exit('The greyscale image has an unexpected format.')

    else:
        logging.warning('There was an error when transforming this image to greyscale.')
        sys.exit('There was an error when transforming this image to greyscale.')
    return image_greyscale


def create_magnitude_image(image: np.ndarray, im_w: int) -> np.ndarray:
    """Create magnitude image."""
    # transform image to greyscale
    magnitude = tranform_to_greyscale(image)

    # resize and normalize magnitude image
    magnitude_resized = resize(magnitude, (im_w, im_w))
    magnitude_resized_normalized = min_max_normalization(magnitude_resized)
    # ensure that magnitude is not zero (leads to strange effects when looking at angle)
    magnitude_resized_normalized[magnitude_resized_normalized == 0] = 1e-9
    return magnitude_resized_normalized


def create_random_phase(im_w: int, rng_seed):
    """Create random phase image."""
    noise = np.zeros((im_w, im_w), dtype=complex)
    noise[
        im_w // 2 - 1 : im_w // 2 + 2,
        im_w // 2 - 1 : im_w // 2 + 2,
    ] = rng_seed.random((3, 3)) + 1j * rng_seed.random((3, 3))
    # create phase image from random k-space data
    random_phase = np.imag(np.fft.ifftn(np.fft.fftshift((noise))))
    # normalize between -pi and pi
    random_phase_norm = min_max_normalization(random_phase) * 2 * np.pi - np.pi
    return random_phase_norm


def create_high_pass_filter_magnitude(magnitude: np.ndarray, im_w: int):
    """Create random high-pass filtered image."""
    k_space = np.fft.fftshift(np.fft.fftn(magnitude))
    # set multiple center pixel to 0 to obtain a  high-pass filtered image
    k_space[
        im_w // 2 - 7 : im_w // 2 + 8,
        im_w // 2 - 7 : im_w // 2 + 8,
    ] = 0
    high_pass_magnitude = np.real(np.fft.ifftn(np.fft.ifftshift(k_space)))
    # normalize between -1 and 1
    high_pass_magnitude_normalized = (
        2 * (min_max_normalization(high_pass_magnitude)) - 1
    )
    return high_pass_magnitude_normalized


def create_phase_image(magnitude: np.ndarray, im_w: int, rng_seed) -> np.ndarray:
    """Create phase image from magnitude image."""
    random_phase = create_random_phase(im_w, rng_seed)

    high_pass_magnitude = create_high_pass_filter_magnitude(magnitude, im_w)

    tot_phase = random_phase + 0.25 * high_pass_magnitude
    return tot_phase


def combine_magnitude_and_phase_image(
    magnitude: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """Create complex-valued image from magnitude and phase image."""
    return magnitude * np.exp(1j * phase)


def create_complex_valued_image(image: np.ndarray, im_w: int, rng_seed) -> np.ndarray:
    """Create complex valued image."""
    # create magnitude image
    magnitude = create_magnitude_image(image, im_w)

    # create phase image
    phase = create_phase_image(magnitude, im_w, rng_seed)

    # combine magnitude and phase image
    complex_valued_image = combine_magnitude_and_phase_image(magnitude, phase)
    return complex_valued_image

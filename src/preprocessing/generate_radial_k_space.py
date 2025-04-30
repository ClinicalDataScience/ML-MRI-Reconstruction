"""Function for building a PyNUFFT object with a radial k-space trajectory."""

import numpy as np
import pynufft


def build_pynufft_object(
    traj,
    im_w: int,
    k_w: int,
    interpolation_w: int,
):
    """Build a PyNUFFT object."""
    Nd = (im_w, im_w)
    Kd = (k_w, k_w)
    Jd = (interpolation_w, interpolation_w)

    NufftObj = pynufft.NUFFT()
    NufftObj.plan(traj, Nd, Kd, Jd)
    return NufftObj

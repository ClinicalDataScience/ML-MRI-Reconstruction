"""Function for building a PyNUFFT object with a radial k-space trajectory."""

import pynufft
from src.utils.trajectory import define_radial_trajectory


def build_pynufft_object(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    k_w: int,
    interpolation_w: int,
):
    """Build a PyNUFFT object."""
    Nd = (im_w, im_w)
    Kd = (k_w, k_w)
    Jd = (interpolation_w, interpolation_w)

    traj = define_radial_trajectory(num_spokes, num_readouts)
    NufftObj = pynufft.NUFFT()
    NufftObj.plan(traj, Nd, Kd, Jd)
    return NufftObj

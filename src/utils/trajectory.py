"""Function for defining a radial k-space trajectory."""
import numpy as np


def define_trajectory(
    num_spokes: int,
    num_readouts: int,
    traj_variant: str = 'radial',
    traj_angle: int = 360,
    traj_shift: int = 0,
    traj_isotropy: int = 1,
) -> np.ndarray:
    """Radial k-space trajectory, normalized from -pi to pi."""
    if traj_variant != 'radial':
        raise ValueError(f'unknown trajectory: {traj_variant}')
    traj = np.zeros((num_spokes * num_readouts, 2))
    vec = np.pi * np.linspace(1, -1, num_readouts, endpoint=False)
    max_angle = np.deg2rad(traj_angle)
    phi0 = max_angle * (np.arange(num_spokes) / num_spokes)
    phi0 += traj_shift * phi0[1]
    if np.isclose(traj_isotropy, 1):
        phi = phi0
    else:
        phi = np.arctan(np.tan(phi0) * traj_isotropy)
        phi[phi < 0] += np.pi
    for n in range(num_spokes):
        traj[n * num_readouts : (n + 1) * num_readouts, 0] = vec * np.cos(phi[n])
        traj[n * num_readouts : (n + 1) * num_readouts, 1] = vec * np.sin(phi[n])
    print(
        f'Defined {traj_variant} trajectory with max. angle {traj_angle}, shift {traj_shift} and isotropy {traj_isotropy}'
    )
    return traj

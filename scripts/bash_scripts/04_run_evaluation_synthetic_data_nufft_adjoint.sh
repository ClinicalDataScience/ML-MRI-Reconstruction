#!/bin/bash

export config_name=$1
export num_spokes=$2

export reconstruction_method='nufft_adjoint'
export reconstruction_device='cpu'

python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --device $reconstruction_device --subfolder_name 'standard' --warm_up

python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --device $reconstruction_device --subfolder_name 'noise' --warm_up

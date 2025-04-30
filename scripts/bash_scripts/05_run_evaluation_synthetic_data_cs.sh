#!/bin/bash

export config_name=$1
export num_spokes=$2
export reconstruction_method='CS'

export bart_regularization=1e-5
export bart_regularization_option='l1'
export bart_maxiter=2000


list_reconstruction_devices=( 'cpu'
                              'cuda'
)


for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
    python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --maxiter $bart_maxiter --device $reconstruction_device --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --subfolder_name 'standard' --warm_up
done

python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --maxiter $bart_maxiter --device 'cpu' --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --subfolder_name 'noise' --warm_up

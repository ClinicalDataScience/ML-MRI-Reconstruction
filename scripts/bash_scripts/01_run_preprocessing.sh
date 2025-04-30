#!/bin/bash

export config_name=$1
export num_spokes=$2

python -m scripts.02_preprocessing_input --config_file_name $config_name --num_spokes $num_spokes --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --subfolder_name 'standard' --save_for_bart_reconstruction --generate_k_space_for_train_set --generate_k_space_for_validation_set --generate_k_space_for_test_set

python -m scripts.02_preprocessing_input --config_file_name $config_name --num_spokes $num_spokes --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --subfolder_name 'noise' --noise --save_for_bart_reconstruction --generate_k_space_for_test_set

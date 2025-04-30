#!/bin/bash

export config_name=$1
export num_spokes=$2
export reconstruction_method='ML'


list_reconstruction_devices=( 'cpu'
                              'cuda'
)

for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
    python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --folder_name_ml_model $folder_name_ml_model --subfolder_name 'standard' --model_name $model_name --device $reconstruction_device  --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --warm_up
done


python -m scripts.04_evaluation_synthetic_data --config_file_name $config_name --num_spokes $num_spokes --method $reconstruction_method --folder_name_ml_model $folder_name_ml_model --subfolder_name 'noise' --model_name $model_name --device 'cpu'  --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --warm_up

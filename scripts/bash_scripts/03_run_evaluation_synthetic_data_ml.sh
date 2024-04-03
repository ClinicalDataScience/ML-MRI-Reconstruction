#!/bin/bash

export config_name=$1
export num_spokes=$2
export reconstruction_method='ML'
export folder_name_ml_model='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export num_spokes_dropout=1

list_reconstruction_devices=( 'cpu'
                              'cuda'
)

export subfolder_name='standard'
for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
    python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --folder_name_ml_model $folder_name_ml_model --subfolder_name $subfolder_name --model_name $model_name --device $reconstruction_device --num_spokes_dropout $num_spokes_dropout --warm_up
done

export reconstruction_device='cpu'
export subfolder_name_noise='noise'
python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --folder_name_ml_model $folder_name_ml_model --subfolder_name $subfolder_name_noise --model_name $model_name --device $reconstruction_device --num_spokes_dropout $num_spokes_dropout --warm_up

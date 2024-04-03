#!/bin/bash

export config_name=$1
export num_spokes=$2
export reconstruction_method='nufft_adjoint'

list_reconstruction_devices=( 'cpu'
                              'cuda'
)

export subfolder_name='standard'
for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
    python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --device $reconstruction_device --subfolder_name $subfolder_name --warm_up

done

export reconstruction_device='cpu'
export subfolder_name_noise='noise'
python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --device $reconstruction_device --subfolder_name $subfolder_name_noise --warm_up

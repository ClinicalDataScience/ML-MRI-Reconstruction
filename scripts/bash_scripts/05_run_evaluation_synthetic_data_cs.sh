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



export subfolder_name='standard'
for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
    python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --maxiter $bart_maxiter --device $reconstruction_device --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --subfolder_name $subfolder_name --warm_up
done

export reconstruction_device='cpu'
export subfolder_name_noise='noise'
python -m scripts.05_evaluation_synthetic_data --config_file_name $config_name --num_spokes $number_of_spokes --method $reconstruction_method --maxiter $bart_maxiter --device $reconstruction_device --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --subfolder_name $subfolder_name_noise --warm_up

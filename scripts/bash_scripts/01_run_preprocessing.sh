#!/bin/bash

export config_name=$1
export num_spokes=$2

export subfolder_name='standard'
python -m scripts.02_preprocessing_input --config_file_name $config_name --num_spokes $num_spokes --subfolder_name $subfolder_name
python -m scripts.03_input_normalization_factor --config_file_name $config_name --num_spokes $num_spokes --subfolder_name $subfolder_name

export subfolder_name_noise='noise'
python -m scripts.02_preprocessing_input --config_file_name $config_name --num_spokes $num_spokes --subfolder_name $subfolder_name_noise --noise

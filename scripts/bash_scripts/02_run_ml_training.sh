#!/bin/bash

export config_name=$1
export number_of_spokes=$2
export folder_name_ml_model='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export num_spokes_dropout=1
export subfolder_name='standard'

python -m scripts.04_ml_training --config_file_name $config_name --num_spokes $number_of_spokes --folder_name_ml_model $folder_name_ml_model --model_name $model_name --num_spokes_dropout $num_spokes_dropout --subfolder_name $subfolder_name

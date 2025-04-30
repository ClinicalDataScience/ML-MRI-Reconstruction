#!/bin/bash

export config_name=$1
export number_of_spokes=$2

echo $dropout
echo $noise_level


python -m scripts.03_ml_training --config_file_name $config_name --num_spokes $number_of_spokes --folder_name_ml_model $folder_name_ml_model --model_name $model_name --dropout $dropout --subfolder_name 'standard' --batch_size 128 --noise_level $noise_level  --dropout $dropout

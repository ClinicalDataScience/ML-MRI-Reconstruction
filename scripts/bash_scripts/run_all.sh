#!/bin/bash

export config_name=$1

list_number_of_spokes=( 101
                        67
                        51
                        41
                        33
                        21
)


python -m scripts.01_preprocessing_output --config_file_name $config_name
for number_of_spokes in "${list_number_of_spokes[@]}" ; do
    . scripts/bash_scripts/01_run_preprocessing.sh $config_name $number_of_spokes
    . scripts/bash_scripts/02_run_ml_training.sh $config_name $number_of_spokes
    . scripts/bash_scripts/03_run_evaluation_synthetic_data_ml.sh $config_name $number_of_spokes
    . scripts/bash_scripts/04_run_evaluation_synthetic_data_nufft_adjoint.sh $config_name $number_of_spokes
    . scripts/bash_scripts/05_run_evaluation_synthetic_data_cs.sh $config_name $number_of_spokes
done

. scripts/bash_scripts/06_run_evaluation_mr_measurements.sh $config_name

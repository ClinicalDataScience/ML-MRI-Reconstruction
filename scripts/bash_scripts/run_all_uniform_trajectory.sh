#!/bin/bash

export config_name=$@
echo $config_name

export traj_variant='radial'
export traj_angle=360
export traj_isotropy=1
export traj_shift=0

export folder_name_ml_model='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export dropout=12e-2
export noise_level=2e-1

export bart_regularization=1e-5
export bart_regularization_option='l1'
export bart_maxiter=2000

list_number_of_spokes=( 101
                        67
                        51
                        41
                        33
                        21
)


# generate complex image
python -m scripts.01_preprocessing_output --config_file_name $config_name

# generate radial k-space data
for number_of_spokes in "${list_number_of_spokes[@]}" ; do
    . scripts/bash_scripts/01_run_preprocessing.sh $config_name $number_of_spokes
    . scripts/bash_scripts/02_run_ml_training.sh $config_name $number_of_spokes
done

# evaluation
for number_of_spokes in "${list_number_of_spokes[@]}" ; do
    . scripts/bash_scripts/03_run_evaluation_synthetic_data_ml.sh $config_name $number_of_spokes
    . scripts/bash_scripts/04_run_evaluation_synthetic_data_nufft_adjoint.sh $config_name $number_of_spokes
    . scripts/bash_scripts/05_run_evaluation_synthetic_data_cs.sh $config_name $number_of_spokes
done
. scripts/bash_scripts/06_run_evaluation_mr_measurements.sh $config_name

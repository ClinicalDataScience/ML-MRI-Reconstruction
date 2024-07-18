#!/bin/bash

export config_name=$1
echo $config_name


set_orientation_list=(0
                            1
                            2)

export folder_name_ml_model='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export bart_regularization=1e-5
export bart_regularization_option='l1'
export bart_stepsize=2e-6
export maxiter=2000
export subfolder_name='liver_measurements'
export selected_coils_list='4,5,6,7,8,9,10,11'
export num_repeat=1
export number_spokes=33
export reconstruction_device='cuda'
export filename='filename_33_spokes'

for set_orientation in "${set_orientation_list[@]}" ; do
    for set_image_number in {0..65}; do
        python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes $number_spokes --method 'ML' --device $reconstruction_device --filename $filename --image_number $set_image_number --orientation $set_orientation --model_name $model_name --folder_name_ml_model $folder_name_ml_model  --subfolder_name $subfolder_name --selected_coils_list $selected_coils_list --num_repeat $num_repeat

        python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes $number_spokes --method 'nufft_adjoint' --device $reconstruction_device --filename $filename --image_number $set_image_number --orientation $set_orientation --subfolder_name $subfolder_name --selected_coils_list $selected_coils_list --num_repeat $num_repeat

        python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes $number_spokes --method 'CS' --device $reconstruction_device --filename $filename --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --subfolder_name $subfolder_name --selected_coils_list $selected_coils_list --num_repeat $num_repeat
    done
done

#!/bin/bash

export config_name=$1
export subfolder_name='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export bart_regularization=1e-5
export bart_regularization_option='l1'
export bart_stepsize=1e-6
export maxiter=2000
export set_image_number=5

export filename_101_spokes='filename_101_spokes'
export filename_67_spokes='filename_67_spokes'
export filename_51_spokes='filename_51_spokes'
export filename_41_spokes='filename_41_spokes'
export filename_33_spokes='filename_33_spokes'
export filename_21_spokes='filename_21_spokes'

list_reconstruction_methods=( 'ML'
                              'nufft_adjoint'
                              'CS'
)
list_reconstruction_devices=( 'cpu'
                              'cuda'
)
set_orientation_list=(0
                    1
                    2)

for set_orientation in "${set_orientation_list[@]}" ; do
    for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
        for reconstruction_method in "${list_reconstruction_methods[@]}" ; do
            if [ $reconstruction_method = 'CS' ]; then
                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 101 --method $reconstruction_method --device $reconstruction_device --filename $filename_101_spokes --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 67 --method $reconstruction_method --device $reconstruction_device --filename $filename_67_spokes  --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 51 --method $reconstruction_method --device $reconstruction_device --filename $filename_51_spokes --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 41 --method $reconstruction_method --device $reconstruction_device --filename $filename_41_spokes --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 33 --method $reconstruction_method --device $reconstruction_device --filename $filename_33_spokes --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 21 --method $reconstruction_method --device $reconstruction_device --filename $filename_21_spokes --image_number $set_image_number --orientation $set_orientation --maxiter $maxiter   --bart_regularization_option $bart_regularization_option --bart_regularization $bart_regularization --bart_stepsize $bart_stepsize --warm_up


            elif [ $reconstruction_method = 'ML' ]; then
                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 101 --method $reconstruction_method --device $reconstruction_device --filename $filename_101_spokes --image_number $set_image_number --orientation $set_orientation --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 67 --method $reconstruction_method --device $reconstruction_device --filename $filename_67_spokes  --image_number $set_image_number --orientation $set_orientation   --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 51 --method $reconstruction_method --device $reconstruction_device --filename $filename_51_spokes --image_number $set_image_number --orientation $set_orientation   --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 41 --method $reconstruction_method --device $reconstruction_device --filename $filename_41_spokes --image_number $set_image_number --orientation $set_orientation --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 33 --method $reconstruction_method --device $reconstruction_device --filename $filename_33_spokes --image_number $set_image_number --orientation $set_orientation   --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 21 --method $reconstruction_method --device $reconstruction_device --filename $filename_21_spokes --image_number $set_image_number --orientation $set_orientation   --model_name $model_name --folder_name_ml_model $subfolder_name  --warm_up



            else
                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 101 --method $reconstruction_method --device $reconstruction_device --filename $filename_101_spokes --image_number $set_image_number --orientation $set_orientation  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 67 --method $reconstruction_method --device $reconstruction_device --filename $filename_67_spokes  --image_number $set_image_number --orientation $set_orientation  --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 51 --method $reconstruction_method --device $reconstruction_device --filename $filename_51_spokes --image_number $set_image_number --orientation $set_orientation --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 41 --method $reconstruction_method --device $reconstruction_device --filename $filename_41_spokes --image_number $set_image_number --orientation $set_orientation --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 33 --method $reconstruction_method --device $reconstruction_device --filename $filename_33_spokes --image_number $set_image_number --orientation $set_orientation --warm_up


                python -m scripts.06_reconstruct_measurements --config_file_name $config_name --num_spokes 21 --method $reconstruction_method --device $reconstruction_device --filename $filename_21_spokes --image_number $set_image_number --orientation $set_orientation --warm_up

            fi
        done
    done
done

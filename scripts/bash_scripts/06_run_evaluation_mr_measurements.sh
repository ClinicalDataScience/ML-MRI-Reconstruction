#!/bin/bash

export config_name=$1
echo $config_name

export image_number=5
export subfolder_name='phantom_measurements'

list_number_of_spokes=( 101
                        67
                        51
                        41
                        33
                        21
)

list_reconstruction_devices=( 'cpu'
                              'cuda'
)

set_orientation_list=(0
                      1
                      2
)


for number_spokes in "${list_number_of_spokes[@]}" ; do

    if [ $number_spokes = 101 ]; then
        export filename='filename_101_spokes'
    elif [ $number_spokes = 67 ]; then
        export filename='filename_67_spokes'
    elif [ $number_spokes = 51 ]; then
        export filename='filename_51_spokes'
    elif [ $number_spokes = 41 ]; then
        export filename='filename_41_spokes'
    elif [ $number_spokes = 33 ]; then
        export filename='filename_33_spokes'
    elif [ $number_spokes = 21 ]; then
        export filename='filename_21_spokes'
    fi

    for set_orientation in "${set_orientation_list[@]}" ; do

        if [ $set_orientation = 1 ]; then
            export num_repeat=100
        else
            export num_repeat=1
        fi

    python -m scripts.05_reconstruct_measurements --config_file_name $config_name --num_spokes $number_spokes --method 'nufft_adjoint' --device 'cpu' --filename $filename --image_number $image_number --orientation $set_orientation  --subfolder_name $subfolder_name  --num_repeat $num_repeat --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --warm_up

        for reconstruction_device in "${list_reconstruction_devices[@]}" ; do
            python -m scripts.05_reconstruct_measurements --config_file_name $config_name --num_spokes $number_spokes --method 'ML' --device $reconstruction_device --filename $filename --image_number $image_number --orientation $set_orientation --model_name $model_name --folder_name_ml_model $folder_name_ml_model  --subfolder_name $subfolder_name  --num_repeat $num_repeat --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $traj_shift --traj_isotropy $traj_isotropy --warm_up

        done
    done
done

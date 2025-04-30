#!/bin/bash


export config_name=$1
echo $config_name
export num_spokes=34

export traj_variant='radial'
export traj_angle=180
export traj_isotropy=2

export folder_name_ml_model='LinearFCNetwork_spoke_dropout'
export model_name='LinearFCNetwork'
export dropout=12e-2
export noise_level=2e-1

export num_repeat=1

export name_subfolders_trajshift_base='trajshift'

trajshift_vals=(
    0
    25e-2
    50e-2
    75e-2
)

python -m scripts.01_preprocessing_output --config_file_name $config_name

for trajshift_val in "${trajshift_vals[@]}"; do
    name_subfolders_trajshift="${name_subfolders_trajshift_base}${trajshift_val}"
    echo $name_subfolders_trajshift

    python -m scripts.02_preprocessing_input --config_file_name $config_name --num_spokes $num_spokes --subfolder_name $name_subfolders_trajshift --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $trajshift_val --traj_isotropy $traj_isotropy --generate_k_space_for_train_set --generate_k_space_for_validation_set --generate_k_space_for_test_set --save_for_bart_reconstruction

    python -m scripts.03_ml_training --config_file_name $config_name --num_spokes $num_spokes --folder_name_ml_model $folder_name_ml_model --model_name $model_name --dropout $dropout --batch_size 128 --noise_level $noise_level --subfolder_name $name_subfolders_trajshift --folder_name_ml_model $name_subfolders_trajshift

done

filename_names=(
    #<ADD_FILENAMES_HERE>
)

for filename in "${filename_names[@]}"; do
    for trajshift_val in "${trajshift_vals[@]}"; do
        name_subfolders_trajshift="${name_subfolders_trajshift_base}${trajshift_val}"
        echo $name_subfolders_trajshift
    
        python -m scripts.05_reconstruct_measurements --config_file_name $config_name --num_spokes $num_spokes --method 'ML' --device 'cpu' --filename $filename  --model_name $model_name --folder_name_ml_model $folder_name_ml_model  --subfolder_name $name_subfolders_trajshift --num_repeat $num_repeat --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $trajshift_val --traj_isotropy $traj_isotropy --folder_name_ml_model $name_subfolders_trajshift
    
        python -m scripts.05_reconstruct_measurements --config_file_name $config_name --num_spokes $num_spokes --method 'ML' --device 'cuda' --filename $filename  --model_name $model_name --folder_name_ml_model $folder_name_ml_model  --subfolder_name $name_subfolders_trajshift --num_repeat $num_repeat --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $trajshift_val --traj_isotropy $traj_isotropy --folder_name_ml_model $name_subfolders_trajshift
    
          python -m scripts.05_reconstruct_measurements --config_file_name $config_name --num_spokes $num_spokes --method 'nufft_adjoint' --device 'cpu' --filename $filename  --subfolder_name $name_subfolders_trajshift  --num_repeat $num_repeat  --traj_variant $traj_variant --traj_angle $traj_angle --traj_shift $trajshift_val  --traj_isotropy $traj_isotropy  --folder_name_ml_model $name_subfolders_trajshift
    
    done
done

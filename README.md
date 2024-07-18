# Fast machine learning reconstruction of radially undersampled k-space data for interventional MRI
This repository contains code for the fast machine learning (ML) reconstruction of radially undersampled 2D k-space data acquired during MR-guided percutaneous interventions.

The code has been developed for the paper *"Fast machine learning reconstruction of radially undersampled k-space data for interventional MRI"* by Topalis et al..

## Computing environment/docker container

### Docker hub

The computing environment for this repository can be obtained from docker hub:

`docker pull docker.io/balthasarschachtner/radler:ML-MRI-Reconstruction`

To build the container yourself, go to the `docker` directory and run `docker build .`.

## Run code
This repository contains code for the training and evaluation of the proposed ML model for the fast reconstruction of undersampled MR data.

It provides code for:
1. Generating synthetic data
2. Training the machine learning model
3. Reconstructing synthetic data
4. Reconstructing MR data

Before running the code, adjust paths in a config file with the name `<config_name>` (e.g., `config.json`) in the `configs/` folder.

Download the training and validation images of the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) [1, 2] dataset and move them to the directory specified in `path_to_images` in the config file.

We used functions provided in the [twixtools](https://github.com/pehses/twixtools.git) repository to load MRI raw data.
For this, we  cloned the repository in the folder `src/utils/` (and adjusted import paths if required).

### 1. Generating synthetic data
We used synthetic data for training the ML model and evaluating its reconstruction performance.

First, we generate complex-valued images with similar magnitude and phase characteristics as MR images from natural images (from the ILSVRC2012 dataset).
The images are split into a training set with `train_set_size`, a validation set with `validation_set_size` and a test set with `test_set_size` images.
The size of each set is defined in the config file.
For generating the complex-valued images and performing the data split, we run this script:
```shell
python -m scripts.01_preprocessing_output --config_file_name <config_name>
```
Next, we generate radial k-space data with `<number_of_spokes>` spokes from the complex-valued images:
```shell
 python -m scripts.02_preprocessing_input --config_file_name <config_name> --num_spokes <number_of_spokes> --subfolder_name <subfolder_name>
```
To analyze the robustness of the reconstruction method to noise, we generated k-space data without additional Gaussian noise (set `<subfolder_name>=standard`) and with additional Gaussian noise (set `<subfolder_name>=noise` and set flag `--noise`) in this project.

The normalization factors for the synthetic k-space data required for the ML reconstruction are calculated with
```shell
 python -m scripts.03_input_normalization_factor --config_file_name <config_name> --num_spokes <number_of_spokes> --subfolder_name standard
```
We selected the same normalization factor for the k-space data with and without additional Gaussian noise.

### 2. Training the machine learning model
We train the ML model with the synthetic training data as follows:
```shell
 python -m scripts.04_ml_training --config_file_name <config_name> --num_spokes <number_of_spokes> --device <device> --batch_size <batch_size> --model_name <model_name> --optimizer_name <optimizer_name> --weight_decay <weight_decay> --folder_name_ml_model <folder_name_ml_model> --num_spokes_dropout <num_spokes_dropout> --subfolder_name <subfolder_name>
```
Please note that the following classes required for training the ML model are not included in this code upload:
- a class `EarlyStopping` for stopping the training of the ML model in `src/machine_learning/training/helpers/early_stopping.py`
- a class `SaveBestModel` for saving the model in `src/machine_learning/training/helpers/save_best_model.py`

### 3. Reconstructing synthetic data
Next, we compare the reconstruction performance of the proposed ML model on the synthetic test data
```shell
 python -m scripts.05_evaluation --config_file_name <config_name> --num_spokes <number_of_spokes> --method ML --device <reconstruction_device> --model_name <model_name> --folder_name_ml_model <folder_name_ml_model> --num_spokes_dropout <num_spokes_dropout> --subfolder_name <subfolder_name> --warm_up
```
with the performance of an adjoint NUFFT with zero-filling
```shell
 python -m scripts.05_evaluation --config_file_name <config_name> --num_spokes <number_of_spokes> --method nufft_adjoint --device <reconstruction_device> --subfolder_name <subfolder_name> --warm_up
```
and a compressed sensing (CS) approach
```shell
 python -m scripts.05_evaluation --config_file_name <config_name> --num_spokes <number_of_spokes> --method CS --device <reconstruction_device> --device <reconstruction_device> --bart_regularization_option <bart_regularization_option> --bart_regularization <bart_regularization> --maxiter <bart_maxiter> --subfolder_name <subfolder_name> --warm_up
```
implemented in the Berkeley advanced reconstruction toolbox (BART) [3, 4].

### 4. Reconstructing MR data
We reconstruct phantom and ex vivo k-space measurements with the ML model as follows:
```shell
python -m scripts.06_reconstruct_measurements --config_file_name <config_name> --num_spokes <number_of_spokes> --method ML --device <reconstruction_device> --filename <filename> --image_number <set_image_number> --orientation <set_orientation> --model_name <model_name> --folder_name_ml_model <subfolder_name> --subfolder_name <subfolder_name> --selected_coils_list <selected_coils_list> --num_repeat <num_repeat> --warm_up
```
For comparison, we also reconstruct the MR data with a NUFFT
```shell
python -m scripts.06_reconstruct_measurements --config_file_name <config_name> --num_spokes <number_of_spokes> --method nufft_adjoint --device <reconstruction_device> --filename <filename> --image_number <set_image_number> --orientation <set_orientation> --subfolder_name <subfolder_name> --selected_coils_list <selected_coils_list> --num_repeat <num_repeat> --warm_up
```
and with CS
```shell
python -m scripts.06_reconstruct_measurements --config_file_name <config_name> --num_spokes <number_of_spokes> --method CS --device <reconstruction_device> --filename <filename> --image_number <set_image_number> --orientation <set_orientation> --maxiter <maxiter> --bart_regularization_option <bart_regularization_option> --bart_regularization <bart_regularization> --bart_stepsize <bart_stepsize> --subfolder_name <subfolder_name> --selected_coils_list <selected_coils_list> --num_repeat <num_repeat> --warm_up
```

### Reproduce results from paper
To reproduce the results from the paper, run the following bash script:
```shell
. scripts/bash_scripts/run_all.sh <config_name>
```
Please change the name of the raw MR data files in `scripts/bash_scripts/06_run_evaluation_mr_measurements.sh` (phantom experiment) and  `scripts/bash_scripts/06_run_evaluation_mr_measurements_liver.sh` (ex vivo experiment) first.

## Resources
[1] Russakovsky O, Deng J, Su H, Krause J, Satheesh S, Ma S, et al. Imagenet large scale visual recognition challenge. Int J Comput Vis. 2015;115:211-52.

[2] Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. Imagenet: A large-scale hierarchical image database. In: Proc IEEE Comput Soc Conf Comput Vis Pattern Recognit; 2009. p. 248-55.

[3] Uecker M, Ong F, Tamir JI, Bahri D, Virtue P, Cheng JY, et al. Berkeley advanced reconstruction toolbox. In: Proc. Intl. Soc. Mag. Reson. Med.. vol. 23; 2015. p. 2486.

[4] Uecker M, Holme C, Blumenthal M, Wang X, Tan Z, Scholand N, et al. mrirecon/bart: version 0.7.00. Zenodo. 2021 March. Available from: https://doi.org/10.5281/zenodo.4570601.

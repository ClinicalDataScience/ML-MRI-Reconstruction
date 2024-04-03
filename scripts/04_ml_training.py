"""Train machine learning reconstruction model."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Union

import torch
from src.machine_learning.dataset.custom_dataset import (
    CustomDataset,
    find_normalization_factor,
)
from src.machine_learning.model.model_fc import LinearFCNetwork
from src.machine_learning.training.model_trainer import Trainer
from src.utils import set_seed
from src.utils.dataset import define_samples_in_dataset
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import (
    define_ML_model_folder_name,
    load_config,
    make_directory,
)


def main(
    num_spokes: int,
    num_readouts: int,
    im_w: int,
    device: str,
    path_to_data: Union[str, os.PathLike],
    path_to_normalization_factor_csv: Union[str, os.PathLike],
    path_to_split_csv: Union[str, os.PathLike],
    path_to_save: Union[str, os.PathLike],
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_spokes_dropout: int,
    num_epochs: int,
    model_name: str,
    folder_name_ml_model: str,
    subfolder_name: str,
    optimizer_name: str,
    seed: int,
) -> None:
    """Train machine learning reconstruction model."""
    torch.cuda.empty_cache()

    set_seed.seed_all(seed)

    path_to_save_ML_model = define_ML_model_folder_name(
        path_to_save, 'ML_model', folder_name_ml_model, num_spokes
    )
    make_directory(path_to_save_ML_model)

    if model_name == 'LinearFCNetwork':
        model_fc = LinearFCNetwork(num_spokes, num_readouts, im_w, False)
    else:
        logging.warning('This model is not defined.')
        sys.exit('This model is not defined.')

    model_fc = model_fc.to(device)

    normalization_factor = find_normalization_factor(
        path_to_normalization_factor_csv, num_spokes
    )

    filelist_train = define_samples_in_dataset(path_to_split_csv, 'train')
    filelist_validation = define_samples_in_dataset(path_to_split_csv, 'validation')
    dataset_train = CustomDataset(
        path_to_data, num_spokes, filelist_train, normalization_factor, subfolder_name
    )
    dataset_validation = CustomDataset(
        path_to_data,
        num_spokes,
        filelist_validation,
        normalization_factor,
        subfolder_name,
    )

    trainer = Trainer(
        model_fc,
        num_spokes,
        num_readouts,
        dataset_train,
        dataset_validation,
        device,
        optimizer_name,
        batch_size,
        learning_rate,
        weight_decay,
        num_spokes_dropout,
        num_epochs,
        path_to_save_ML_model,
        seed,
    )
    trainer.train_and_validate(
        dataset_train,
        dataset_validation,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_name', required=True, help='Name of config file')
    parser.add_argument(
        '--num_spokes',
        required=True,
        type=int,
        help='Number of spokes in radial k-space',
    )
    parser.add_argument(
        '--device',
        required=False,
        default='cuda',
        choices=['cpu', 'cuda'],
        type=str,
        help='Device for training',
    )
    parser.add_argument(
        '--batch_size',
        required=False,
        default=256,
        type=int,
        help='Batch size for dataloader',
    )
    parser.add_argument(
        '--model_name',
        required=False,
        default='LinearFCNetwork',
        type=str,
        help='Name of model',
    )
    parser.add_argument(
        '--optimizer_name',
        required=False,
        default='Adam',
        choices=['Adam', 'SGD'],
        type=str,
        help='Name of optimizer',
    )

    parser.add_argument(
        '--weight_decay',
        required=False,
        default=0,
        type=float,
        help='Weight decay',
    )

    parser.add_argument(
        '--folder_name_ml_model',
        required=False,
        default=None,
        type=str,
        help='Name of folder where ML model will be saved',
    )

    parser.add_argument(
        '--num_spokes_dropout',
        required=False,
        default=None,
        type=int,
        help='Number of spokes for dropout during training',
    )

    parser.add_argument(
        '--subfolder_name',
        required=False,
        default='radial_k',
        type=str,
        help='Name of folder where the k-space data should be saved',
    )

    args = parser.parse_args()

    config = load_config('configs/' + args.config_file_name)

    set_up_logging_and_save_args_and_config(
        'ml_training_' + str(args.num_spokes) + '_spokes', args, config
    )

    main(
        num_spokes=args.num_spokes,
        num_readouts=config['num_readouts'],
        im_w=config['im_w'],
        device=args.device,
        path_to_data=Path(config['path_to_data']),
        path_to_normalization_factor_csv=Path(
            config['path_to_normalization_factor_csv']
        ),
        path_to_split_csv=Path(config['path_to_split_csv']),
        path_to_save=Path(config['path_to_save']),
        batch_size=args.batch_size,
        learning_rate=config['learning_rate'],
        weight_decay=args.weight_decay,
        num_spokes_dropout=args.num_spokes_dropout,
        num_epochs=config['num_epochs'],
        model_name=args.model_name,
        folder_name_ml_model=args.folder_name_ml_model,
        subfolder_name=args.subfolder_name,
        optimizer_name=args.optimizer_name,
        seed=config['seed'],
    )

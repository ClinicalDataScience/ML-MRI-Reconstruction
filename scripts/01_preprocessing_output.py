"""Generate complex-valued synthetic images and split data."""
import argparse
import hashlib
import os
from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from src.preprocessing.generate_complex_valued_image import create_complex_valued_image
from src.utils import set_seed
from src.utils.logging_functions import set_up_logging_and_save_args_and_config
from src.utils.save_and_load import (
    list_files_in_directory,
    load_config,
    make_directory,
    save_real_and_imaginary_parts_in_2_channels,
)
from tqdm import tqdm


def create_and_save_complex_valued_output(
    image: np.ndarray, im_w: int, rng_seed
) -> np.ndarray:
    """Create complex valued output and reshape it."""
    complex_valued_image = create_complex_valued_image(image, im_w, rng_seed)
    output_data = save_real_and_imaginary_parts_in_2_channels(complex_valued_image)
    return output_data


def split_dataset(
    train_val_names: List[str],
    test_names: List[str],
    path_to_split_csv: Union[str, os.PathLike],
    train_set_size: int,
    validation_set_size: int,
    seed: int,
) -> None:
    """Split dataset."""
    train_names, validation_names = train_test_split(
        train_val_names,
        train_size=train_set_size,
        test_size=validation_set_size,
        random_state=seed,
    )
    assert (
        len(train_names) == train_set_size
    ), f'{len(train_names)} samples in train set instead of {train_set_size}.'
    assert (
        len(validation_names) == validation_set_size
    ), f'{len(validation_names)} samples in validation set instead of {validation_set_size}.'

    make_directory(path_to_split_csv)
    pd.DataFrame(train_names).to_csv(
        os.path.join(path_to_split_csv, 'train_samples.csv'), index=False
    )
    pd.DataFrame(validation_names).to_csv(
        os.path.join(path_to_split_csv, 'validation_samples.csv'), index=False
    )
    pd.DataFrame(test_names).to_csv(
        os.path.join(path_to_split_csv, 'test_samples.csv'), index=False
    )


def generate_complex_valued_output_for_full_dataset(
    hash_dict: Dict[str, str],
    filelist_dataset: List[Union[str, os.PathLike]],
    path_to_dataset: Union[str, os.PathLike],
    path_to_ground_truth: Union[str, os.PathLike],
    dataset_size: int,
    im_w: int,
    rng_seed,
) -> Tuple[List[str], Dict[str, str]]:
    """Generate complex valued output for full dataset."""
    dataset_names = []
    for filename in tqdm(filelist_dataset):
        directory = os.path.join(path_to_dataset, filename)
        image = mpimg.imread(directory)

        h = hashlib.md5(image).hexdigest()
        if h in hash_dict:
            print(f'Skipped image {filename}.')
            continue
        hash_dict[h] = filename

        output_data = create_and_save_complex_valued_output(image, im_w, rng_seed)
        if np.isnan(np.sum(output_data)) == True:
            continue
        file_path_ground_truth = os.path.join(
            path_to_ground_truth, os.path.splitext(filename)[0]
        )
        np.save(file_path_ground_truth, output_data)

        dataset_names.append(os.path.splitext(filename)[0] + '.npy')
        if len(dataset_names) == dataset_size:
            break
    return dataset_names, hash_dict


def main(
    im_w: int,
    train_set_size: int,
    validation_set_size: int,
    test_set_size: int,
    path_to_images: Union[str, os.PathLike],
    path_to_data: Union[str, os.PathLike],
    path_to_split_csv: Union[str, os.PathLike],
    seed: int,
):
    """Generate complex-valued synthetic data from natural images."""
    set_seed.seed_all(seed)
    rng_seed = np.random.default_rng(seed)
    hash_dict = dict()  #  type: Dict[str, str]

    path_to_ground_truth = Path(os.path.join(path_to_data, 'complex_image'))
    make_directory(path_to_ground_truth)

    # select files
    path_to_train_val_images = Path(os.path.join(path_to_images, 'train/'))
    path_to_test_images = Path(os.path.join(path_to_images, 'val/'))

    # data for train and validation set
    if os.path.exists(path_to_train_val_images):
        filelist_train_val = list_files_in_directory(path_to_train_val_images, 'JPEG')
    else:
        filelist_train_val = []

    if len(filelist_train_val) > 0:
        train_val_names, hash_dict = generate_complex_valued_output_for_full_dataset(
            hash_dict,
            filelist_train_val,
            path_to_train_val_images,
            path_to_ground_truth,
            train_set_size + validation_set_size,
            im_w,
            rng_seed,
        )
    else:
        train_val_names = []

    # data for test set
    if os.path.exists(path_to_test_images):
        filelist_test = list_files_in_directory(path_to_test_images, 'JPEG')

        start_image_in_test_set = 1500
        if len(filelist_test) >= start_image_in_test_set:
            for filename in tqdm(filelist_test[:start_image_in_test_set]):
                directory = os.path.join(path_to_test_images, filename)
                image = mpimg.imread(directory)
                h = hashlib.md5(image).hexdigest()
                hash_dict[h] = filename
            filelist_test = filelist_test[start_image_in_test_set:]
        else:
            filelist_test = filelist_test
    else:
        filelist_test = list_files_in_directory(path_to_images, 'JPEG')

    if len(filelist_test) > 0:
        test_names, hash_dict = generate_complex_valued_output_for_full_dataset(
            hash_dict,
            filelist_test,
            path_to_test_images,
            path_to_ground_truth,
            test_set_size,
            im_w,
            rng_seed,
        )
    else:
        test_names = []

    split_dataset(
        train_val_names,
        test_names,
        path_to_split_csv,
        train_set_size,
        validation_set_size,
        seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_name', required=True, help='Name of config file')
    args = parser.parse_args()

    config = load_config('configs/' + args.config_file_name)

    set_up_logging_and_save_args_and_config('preprocessing_output', args, config)

    main(
        im_w=config['im_w'],
        train_set_size=config['train_set_size'],
        validation_set_size=config['validation_set_size'],
        test_set_size=config['test_set_size'],
        path_to_images=Path(config['path_to_images']),
        path_to_data=Path(config['path_to_data']),
        path_to_split_csv=Path(config['path_to_split_csv']),
        seed=config['seed'],
    )

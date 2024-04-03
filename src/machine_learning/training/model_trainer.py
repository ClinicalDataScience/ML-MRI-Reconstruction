"""Class and functions for training and validating a neural network for the reconstruction of radial k-space data."""
import logging
import os
import random
import sys
from typing import Union

import numpy as np
import torch
from src.machine_learning.inference.ml_reconstruction_inference_utils import (
    calculate_normalization_factor,
)
from src.machine_learning.training.helpers.early_stopping import EarlyStopping
from src.machine_learning.training.helpers.save_best_model import SaveBestModel
from src.utils import set_seed
from src.utils.time_measurements import select_timer
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer for machine learning reconstruction model."""

    def __init__(
        self,
        model,
        num_spokes: int,
        num_readouts: int,
        dataset_train,
        dataset_validation,
        device: str,
        optimizer_name: str,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        num_spokes_dropout: int,
        num_epochs: int,
        path_to_save_ML_model: Union[str, os.PathLike],
        seed: int,
    ) -> None:
        """Initialize Trainer."""
        self.model = model
        self.num_spokes = num_spokes
        self.num_readouts = num_readouts
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.num_spokes_dropout = num_spokes_dropout
        if self.num_spokes_dropout is not None:
            self.normalization_spoke_dropout = calculate_normalization_factor(
                self.num_spokes, self.num_spokes_dropout
            )

        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), self.learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate
            )
        else:
            logging.warning('This optimizer is not defined.')
            sys.exit('This optimizer is not defined.')

        self.loss_metric = MSELoss()
        self.num_epochs = num_epochs
        self.path_to_save_ML_model = path_to_save_ML_model
        self.seed = seed

    def spoke_dropout(
        self,
        X: Tensor,
    ) -> Tensor:
        """Set a few random spokes in input tensor to zero."""
        # set random spokes in each batch to zero
        for j in range(X.shape[0]):
            # define which spokes should be set to zero
            a = random.sample(range(0, self.num_spokes), self.num_spokes_dropout)
            for i in range(len(a)):
                X[j, :, a[i] * self.num_readouts : (a[i] + 1) * self.num_readouts] = 0
        return X

    def calculate_loss(self, y, predictions):
        """Calculate loss between ground truth and prediction."""
        loss = self.loss_metric(y, predictions)
        return loss

    def train_step(
        self,
        X: Tensor,
        y: Tensor,
    ) -> float:
        """Do a train step."""
        X = X.to(self.device)
        y = y.to(self.device)

        # set the k-space values of multiple spokes to 0
        if self.num_spokes_dropout is not None:
            X = self.spoke_dropout(X)

        predictions = self.model(X)
        loss = self.calculate_loss(y, predictions)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(
        self,
        X: Tensor,
        y: Tensor,
    ) -> float:
        """Validate model."""
        X = X.to(self.device)
        y = y.to(self.device)

        # normalize input for validation
        if self.num_spokes_dropout is not None:
            X = X * self.normalization_spoke_dropout

        predictions = self.model(X)
        loss = self.loss_metric(y, predictions)
        return loss.item()

    def save_loss(
        self,
        train_loss_numpy: np.ndarray,
        validation_loss_numpy: np.ndarray,
    ) -> None:
        """Save training and validation loss."""
        np.save(
            os.path.join(
                self.path_to_save_ML_model,
                'ML_' + str(self.num_spokes) + '_spokes' + '_train_loss',
            ),
            train_loss_numpy,
        )
        np.save(
            os.path.join(
                self.path_to_save_ML_model,
                'ML_' + str(self.num_spokes) + '_spokes' + '_validation_loss',
            ),
            validation_loss_numpy,
        )

    def train_and_validate(
        self,
        dataset_train,
        dataset_validation,
    ) -> None:
        """Train and validate model."""
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            worker_init_fn=set_seed.seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )
        dataloader_validation = DataLoader(
            dataset_validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            worker_init_fn=set_seed.seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

        save_best_model = SaveBestModel()
        early_stopping = EarlyStopping()

        train_loss_epoch_list = []
        validation_loss_epoch_list = []

        timer = select_timer(self.device)

        with timer:
            for epoch in tqdm(range(self.num_epochs)):
                train_loss_add = 0
                validation_loss_add = 0

                # training
                self.model.train()
                for batch, (X, y) in enumerate(dataloader_train):
                    train_loss = self.train_step(X, y)
                    train_loss_add += train_loss * X.shape[0]

                train_loss_epoch = train_loss_add / len(dataloader_train.dataset)
                train_loss_epoch_list.append(train_loss_epoch)

                # validation
                self.model.eval()
                with torch.no_grad():
                    for batch, (X, y) in enumerate(dataloader_validation):
                        validation_loss = self.validation_step(X, y)
                        validation_loss_add += validation_loss * X.shape[0]

                validation_loss_epoch = validation_loss_add / len(
                    dataloader_validation.dataset
                )
                validation_loss_epoch_list.append(validation_loss_epoch)

                logging.info(
                    'Epoch {}: Train loss: {}, Validation loss: {}'.format(
                        epoch, train_loss_epoch, validation_loss_epoch
                    )
                )

                save_best_model(
                    validation_loss_epoch,
                    epoch,
                    self.model,
                    self.optimizer,
                    self.path_to_save_ML_model,
                    self.num_spokes,
                )

                self.save_loss(
                    np.array(train_loss_epoch_list),
                    np.array(validation_loss_epoch_list),
                )

                early_stopping(validation_loss_epoch)
                if early_stopping.early_stop:
                    break
        logging.info('Training and Validation time [s]: %s', timer.execution_time)

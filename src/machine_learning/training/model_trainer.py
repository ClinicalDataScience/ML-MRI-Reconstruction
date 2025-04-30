"""Class and functions for training and validating a neural network for the reconstruction of radial k-space data."""
import logging
import os
import random
import sys
from typing import Union

import numpy as np
import torch
from src.machine_learning.training.helpers.early_stopping import EarlyStopping
from src.machine_learning.training.helpers.save_best_model import SaveBestModel
from src.utils import set_seed
from src.utils.time_measurements import select_timer
from torch import Tensor
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        dropout: int,
        num_epochs: int,
        path_to_save_ML_model: Union[str, os.PathLike],
        noise_level: float,
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
        self.dropout = dropout

        self.n_dropout = self.calculate_num_spokes_dropout()
        if self.n_dropout != 0:
            print(f'Number of spokes used for dropout {self.n_dropout}')
            self.normalization_spoke_dropout = self.calculate_normalization_factor()

        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.learning_rate,
                betas=(0.9, 0.98),
                weight_decay=weight_decay,
                eps=1e-09,
            )
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate
            )
        else:
            logging.warning('This optimizer is not defined.')
            sys.exit('This optimizer is not defined.')

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.8, eps=1e-10
        )
        self.loss_metric = MSELoss()
        self.num_epochs = num_epochs
        self.path_to_save_ML_model = path_to_save_ML_model
        self.seed = seed
        self.noise_level = noise_level

    def calculate_num_spokes_dropout(self):
        """Calculate number of spokes for dropout."""
        if self.dropout > 0:
            if self.dropout >= 1:  # use as absolute number of spokes to drop
                n_dropout = int(np.round(self.dropout))
            else:  # use as fraction of spokes to drop (but at least 1)
                n_dropout = max(1, int(np.round(self.dropout * self.num_spokes)))
            n_dropout = min(self.num_spokes - 1, n_dropout)
        else:
            n_dropout = 0
        return n_dropout

    def calculate_normalization_factor(self):
        """Calculate normalization factor for the input data during training for the reconstructions with the machine learning model."""
        normalization_spoke_dropout = (
            self.num_spokes - self.n_dropout
        ) / self.num_spokes
        return normalization_spoke_dropout

    def spoke_dropout(
        self,
        X: Tensor,
    ) -> Tensor:
        """Set a few random spokes in input tensor to zero."""
        for j in range(X.shape[0]):
            # define which spokes should be set to zero
            a = random.sample(range(0, self.num_spokes), self.n_dropout)
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

        if self.noise_level != 0:
            uniform_dist = torch.distributions.uniform.Uniform(
                1.0 - self.noise_level, 1.0 + self.noise_level
            )
            X = torch.multiply(X, uniform_dist.sample(X.shape).to(self.device))

        # set the k-space values of multiple spokes to 0
        if self.n_dropout != 0:
            X = self.spoke_dropout(X)
            y *= self.normalization_spoke_dropout

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
                self.scheduler.step(validation_loss_epoch)

                logging.info(
                    'Epoch {}: Train loss: {}, Validation loss: {}, Current learning rate: {}'.format(
                        epoch + 1,
                        train_loss_epoch,
                        validation_loss_epoch,
                        self.optimizer.param_groups[0]['lr'],
                    )
                )
                print(
                    f'Epoch {epoch + 1}: Train loss: {train_loss_epoch}, Validation loss: {validation_loss_epoch}'
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

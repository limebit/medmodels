"""Trainer for Real-GRU."""

import json
import logging
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from medmodels.data_synthesis.mtgan.model.gru.loss import PredictionLoss
from medmodels.data_synthesis.mtgan.model.gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
)
from medmodels.data_synthesis.mtgan.train.gan_trainer import (
    TrainingHyperparametersTotal,
)


class RealGRUTrainer:
    """Trainer for Real-GRU."""

    def __init__(
        self,
        real_gru: RealGRU,
        train_loader: MTGANDataLoader,
        hyperparameters: TrainingHyperparametersTotal,
    ) -> None:
        """Constructor for RealGRUTrainer.

        Args:
            real_gru (RealGRU): Real-GRU model to train
            train_loader (MTGANDataLoader): steps (int), dataset
            hyperparameters (TrainingHyperparametersTotal): hyperparameters dictionary
        """
        self.real_gru = real_gru
        self.train_loader = train_loader

        real_gru_lr = hyperparameters.get("real_gru_lr")
        self.epochs = hyperparameters.get("real_gru_training_epochs")

        self.optimizer = torch.optim.adam.Adam(real_gru.parameters(), lr=real_gru_lr)
        self.loss_fn = PredictionLoss()

    def log_loss_json(self, loss_value: float, epoch: int) -> None:
        """Log loss in JSON format.

        Args:
            loss_value (float): loss value
            epoch (int): epoch number
        """
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"),
            "loss": loss_value,
            "epoch": epoch,
        }
        logging.info(json.dumps(log_entry))

    def train(self) -> None:
        """Training.

        Backpropragtion with Adams-algorithm (extension to stochastic gradient descent)
        and real_gru_epochs epochs."""
        loss = torch.Tensor([np.inf])

        for epoch in tqdm(
            range(1, self.epochs + 1), desc="Training RealGRU", total=self.epochs
        ):
            for _, data in enumerate(self.train_loader, start=1):
                real_data, prediction_data, real_number_admissions = data
                output = self.real_gru(real_data)
                loss: torch.Tensor = self.loss_fn(
                    output, prediction_data, real_number_admissions
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.log_loss_json(loss.item(), epoch)

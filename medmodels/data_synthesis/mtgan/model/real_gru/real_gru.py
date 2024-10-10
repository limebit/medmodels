"""RealGRU for calculating the hidden state for real data.

This pre-training step is used to calculate the hidden states for the real data in the
dataset."""

import numpy as np
import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparameters,
)
from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader
from medmodels.data_synthesis.mtgan.model.masks import find_sequence_mask
from medmodels.data_synthesis.mtgan.model.real_gru.loss import PredictionLoss


class RealGRU(nn.Module):
    total_number_of_concepts: int
    epochs: int
    device: torch.device

    gru: nn.GRU
    linear_layer: nn.Sequential
    optimizer: torch.optim.optimizer.Optimizer
    loss_function: nn.Module

    def __init__(
        self,
        total_number_of_concepts: int,
        hyperparamaters: TrainingHyperparameters,
        device: torch.device,
    ) -> None:
        """Constructor for the RealGRU.

        Args:
            total_number_of_concepts (int): The total number of concepts in the
                dataset.
            hyperparameters (TrainingHyperparameters): The hyperparameters for the
                training.
            device (torch.device): The device to use for the model.
        """
        super().__init__()
        self.total_number_of_concepts = total_number_of_concepts
        self.epochs = hyperparamaters["real_gru_training_epochs"]
        self.device = device
        self.to(self.device)

        self.gru = nn.GRU(
            input_size=total_number_of_concepts,
            hidden_size=hyperparamaters["generator_hidden_dimension"],
            batch_first=True,
            dtype=torch.float32,
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(
                hyperparamaters["generator_hidden_dimension"], total_number_of_concepts
            ),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.adam.Adam(
            self.parameters(), lr=hyperparamaters["real_gru_learning_rate"]
        )
        self.loss_function = PredictionLoss(device=self.device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass of RealGRU computing the predictions for the next window.

        Linear layer for first hidden state, then GRU.

        Args:
            data (torch.Tensor): input data, of shape (batch size, maximum number of
                windows -1, total number of concepts). We take out the last window
                because we are predicting the next window.

        Returns:
            torch.Tensor: Predictions of the RealGRU, of shape (batch size, maximum
                number of windows -1, total number of concepts)
        """
        outputs, _ = self.gru(data)
        return self.linear_layer(outputs)

    def calculate_hidden_states(
        self, data: torch.Tensor, number_of_windows_per_patient: torch.Tensor
    ) -> torch.Tensor:
        """Calculate hidden states with respect to the data and mask them according to the number of windows per patient.

        Args:
            data (torch.Tensor): input data, of shape (batch size, maximum number of
                windows -1, total number of concepts). We take out the last window
                because we are predicting the next window.
            number_of_windows_per_patient (torch.Tensor): number of windows per
                patient

        Returns:
            torch.Tensor: hidden states of the RealGRU with respect to the data, masked
                of shape (batch size, maximum number of windows -1, total number of
                concepts)
        """
        with torch.no_grad():
            maximum_number_of_windows = data.shape[1]
            mask = find_sequence_mask(
                number_of_windows_per_patient, maximum_number_of_windows
            ).unsqueeze(dim=-1)
            hidden_states, _ = self.gru(data)
            return hidden_states * mask

    def train_real_gru(self, train_loader: MTGANDataLoader) -> float:
        """Training the RealGRU model.

        Args:
            train_loader (MTGANDataLoader): The data loader for the training data.

        Returns:
            float: The loss of the model.
        """
        torch.manual_seed(0)
        loss = torch.Tensor([np.inf])

        for _ in range(self.epochs):
            for data in train_loader:
                real_data, real_prediction_data, number_of_windows_per_patient = data
                synthetic_prediction_data = self(real_data)
                loss: torch.Tensor = self.loss_function(
                    real_prediction_data,
                    synthetic_prediction_data,
                    number_of_windows_per_patient,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()
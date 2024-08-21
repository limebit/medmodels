"""Trainer for Real-GRU."""

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
)
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.train.gan_trainer import (
    TrainingHyperparametersTotal,
)


class RealGRUTrainer:
    """Trainer for Real-GRU."""

    real_gru: RealGRU
    train_loader: MTGANDataLoader

    epochs: int
    optimizer: torch.optim.optimizer.Optimizer
    loss_function: nn.Module

    def __init__(
        self,
        real_gru: RealGRU,
        train_loader: MTGANDataLoader,
        hyperparameters: TrainingHyperparametersTotal,
    ) -> None: ...
    def train(self) -> None: ...

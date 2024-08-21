"""Module for training the Critic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
)
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.train.gan_trainer import (
        TrainingHyperparametersTotal,
    )

class CriticTrainer(nn.Module):
    """Class for training the critic."""

    critic: Critic
    generator: Generator
    real_gru: RealGRU
    device: torch.device

    batch_size: int
    critic_iterations: int
    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_function: nn.Module

    def __init__(
        self,
        critic: Critic,
        generator: Generator,
        real_gru: RealGRU,
        hyperparameters: TrainingHyperparametersTotal,
        device: Union[torch.device, torch.cuda.device],
    ) -> None: ...
    def _step(
        self,
        real_data: torch.Tensor,
        number_admissions: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> Tuple[float, float]: ...
    def step(
        self,
        real_data: torch.Tensor,
        real_number_admissions: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> Tuple[float, float]: ...
    def evaluate(
        self,
        data_loader: MTGANDataLoader,
    ) -> float: ...

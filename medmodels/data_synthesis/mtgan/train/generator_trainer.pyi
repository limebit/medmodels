"""Generator Trainer for MTGAN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.train.gan_trainer import (
        TrainingHyperparametersTotal,
    )

class GeneratorTrainer(nn.Module):
    """Trainer for Generator."""

    generator: Generator
    critic: Critic
    device: torch.device

    batch_size: int
    generator_iterations: int
    number_codes: int
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        hyperparameters: TrainingHyperparametersTotal,
    ) -> None: ...
    def _step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float: ...
    def step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float: ...

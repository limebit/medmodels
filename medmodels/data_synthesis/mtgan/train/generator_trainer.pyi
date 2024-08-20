"""Generator Trainer for MTGAN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic

# from torch.nn.utils import clip_grad_norm_
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.train.gan_trainer import (
        TrainingHyperparametersTotal,
    )

class GeneratorTrainer:
    """Trainer for Generator."""

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

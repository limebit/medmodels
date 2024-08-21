from typing import Optional, Tuple

import sparse
import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.model.generator.generator_layers import (
    GRU,
    SmoothAttention,
)

class Generator(nn.Module):
    number_codes: int
    max_number_admissions: int
    device: torch.device

    hidden_dimension: int
    attention_dimension: int
    batch_size: int
    generator_iterations: int
    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    gru: GRU
    smooth_condition: SmoothAttention

    def __init__(
        self,
        number_codes: int,
        max_number_admissions: int,
        hyperparameters: TrainingHyperparametersTotal,
        device: torch.device,
    ) -> None: ...
    def forward(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def generate_data_matrix(
        self,
        number_patients: int,
        windows_distribution: torch.Tensor,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
    ) -> sparse.COO: ...
    def train(
        self,
        critic: Critic,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float: ...

"""Critic for MTGAN.

This module contains the Critic class, which is used in the MTGAN model.
"""

import torch
from torch import nn

class Critic(nn.Module):
    """Critic for the MTGAN model."""

    def __init__(
        self,
        number_codes: int,
        critic_hidden_dimension: int,
        generator_hidden_dimension: int,
    ) -> None: ...
    def forward(
        self, data: torch.Tensor, hiddens: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor: ...

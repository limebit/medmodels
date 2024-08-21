"""RealGRU for calculating the hidden state for Real samples.

This pre-training step is used to calculate the hidden state for the real samples
in the dataset. The hidden state is calculated using a GRU network."""

import torch
from torch import nn


class RealGRU(nn.Module):
    """RealGRU for calculating the hidden state for real samples."""

    number_codes: int
    gru: nn.GRU
    linear: nn.Sequential

    def __init__(self, number_codes: int, real_gru_hidden_dimension: int) -> None: ...
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...
    def calculate_hidden(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor: ...

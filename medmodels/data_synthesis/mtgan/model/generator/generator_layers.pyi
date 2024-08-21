from typing import Optional, Tuple

import torch
from torch import nn

class GRU(nn.Module):
    hidden_dimension: int
    max_number_admissions: int
    device: torch.device

    gru_cell: nn.GRUCell
    hidden2codes: nn.Sequential

    def __init__(
        self,
        number_codes: int,
        gru_hidden_dim: int,
        max_number_admissions: int,
        device: Optional[torch.device] = None,
    ) -> None: ...
    def step(
        self, data: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def forward(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

class AttentionScore(nn.Module):
    number_codes: int
    attention_dimension: int

    def __init__(self, number_codes: int, attention_dim: int) -> None: ...
    def forward(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor: ...

class SmoothAttention(nn.Module):
    atention: AttentionScore

    def __init__(self, number_codes: int, attention_dim: int) -> None: ...
    def forward(
        self,
        probability_matrix: torch.Tensor,
        number_windows: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> torch.Tensor: ...

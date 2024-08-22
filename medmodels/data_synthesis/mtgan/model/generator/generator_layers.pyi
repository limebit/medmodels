from typing import Optional, Tuple

import torch
from torch import nn

class GRU(nn.Module):
    hidden_dimension: int
    max_number_windows: int
    device: torch.device

    gru_cell: nn.GRUCell
    linear_layer: nn.Sequential

    def __init__(
        self,
        total_number_of_concepts: int,
        hidden_dimension: int,
        max_number_windows: int,
        device: Optional[torch.device] = None,
    ) -> None: ...
    def step(
        self, data: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def forward(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

class AttentionScore(nn.Module):
    total_number_of_concepts: int
    attention_dimension: int

    def __init__(self, total_number_of_concepts: int, attention_dim: int) -> None: ...
    def forward(
        self, data: torch.Tensor, number_windows: torch.Tensor
    ) -> torch.Tensor: ...

class SmoothAttention(nn.Module):
    atention: AttentionScore

    def __init__(self, total_number_of_concepts: int, attention_dim: int) -> None: ...
    def forward(
        self,
        probability_matrix: torch.Tensor,
        number_windows: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> torch.Tensor: ...

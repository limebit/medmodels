import torch
from torch import nn

class RealGRU(nn.Module):
    number_codes: int
    gru: nn.GRU
    linear: nn.Sequential

    def __init__(self, number_codes: int, real_gru_hidden_dimension: int) -> None: ...
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...
    def calculate_hidden(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor: ...

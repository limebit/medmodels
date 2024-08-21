from typing import Optional, Tuple, Union

import sparse
import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.generator.generator_layers import (
    GRU,
    SmoothAttention,
)


class Generator(nn.Module):
    number_codes: int
    max_number_admissions: int
    hidden_dimension: int
    attention_dimension: int
    device: torch.device

    gru: GRU
    smooth_condition: SmoothAttention

    def __init__(
        self,
        number_codes: int,
        max_number_admissions: int,
        hidden_dimension: int,
        attention_dimension: int,
        device: Optional[torch.device] = None,
    ) -> None: ...
    def forward(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def sample(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
    def sample_with_hidden_states(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def get_noise(self, batch_size: int) -> torch.Tensor: ...
    def get_target_codes(self, batch_size: int) -> torch.Tensor: ...
    def generate_data_matrix(
        self,
        number_patients: int,
        windows_distribution: torch.Tensor,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
    ) -> sparse.COO: ...
    def get_required_number(
        self,
        windows_distribution: torch.Tensor,
        batch_size: int,
        upper_bound: int,
    ) -> Union[bool, int]: ...

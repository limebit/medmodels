"""RealGRU for calculating the hidden state for Real samples.

This pre-training step is used to calculate the hidden state for the real samples
in the dataset. The hidden state is calculated using a GRU network."""

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import sequence_mask


class RealGRU(nn.Module):
    """RealGRU for calculating the hidden state for real samples."""

    def __init__(self, number_codes: int, real_gru_hidden_dimension: int) -> None:
        """Constructor for the RealGRU.

        Args:
            number_codes (int): Number of codes in the dataset.
            real_gru_hidden_dimension (int): Hidden dimension of the RealGRU.
        """
        super(RealGRU, self).__init__()
        self.number_codes = number_codes
        self.gru = nn.GRU(
            input_size=number_codes,
            hidden_size=real_gru_hidden_dimension,
            batch_first=True,
            dtype=torch.float32,
        )
        self.linear = nn.Sequential(
            nn.Linear(real_gru_hidden_dimension, number_codes), nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass of GRU. Linear layer for first hidden state, then GRU.

        Args:
            x (torch.Tensor): input data
        Returns:
            torch.Tensor: output
        """
        outputs, _ = self.gru(data)
        return self.linear(outputs)

    def calculate_hidden(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor:
        """Calculate hidden states with and apply mask.

        Args:
            data (torch.Tensor): input data
            number_admissions (torch.Tensor): number of admissions per patient

        Returns:
            torch.Tensor: hidden state of the GRU, masked
        """
        with torch.no_grad():
            maximum_num_admissions = data.shape[1]
            mask = sequence_mask(number_admissions, maximum_num_admissions).unsqueeze(
                dim=-1
            )
            outputs, _ = self.gru(data)
            return outputs * mask

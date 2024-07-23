"""Generator Layers.

Equations and sections refer to the paper:
"Multi-level Generative Adversarial Networks for Synthetic Electronic Health Record Generation" by Chang Lu et al. 2023.

This module contains the layers used in the Generator of MTGAN:
    - GRU
    - AttentionScore
    - SmoothCondition
"""

from typing import Optional, Tuple

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import masked_softmax, sequence_mask


class GRU(nn.Module):
    """GRU for generator."""

    def __init__(
        self,
        number_codes: int,
        gru_hidden_dim: int,
        max_number_admissions: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Constructor of the GRU.

        When instantiated, create
            - self.gru_cell: GRUCell for the generator.
            - self.hidden2codes: Sequential layer to convert hidden state to codes.

        Args:
            number_codes (int): Number of codes in the dataset.
            gru_hidden_dim (int): Hidden dimension of the GRU.
            max_number_admissions (int): Maximum number of admissions.
            device (torch.device): Device to run the model.
        """
        super().__init__()
        self.hidden_dim = gru_hidden_dim
        self.max_number_admissions = max_number_admissions
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 4.4.1/ equation 13
        self.gru_cell = nn.GRUCell(input_size=number_codes, hidden_size=gru_hidden_dim)
        # 4.4.1/ equation 14
        self.hidden2codes = nn.Sequential(
            nn.Linear(gru_hidden_dim, number_codes), nn.Sigmoid()
        )

    def step(
        self, data: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step function for the GRU.

        This function is used to compute the next step of the GRU by performing the
        following operations:
            - Update the hidden states for the current time step.
            - Transforms the updated hidden states into the output concept codes.

        Args:
            data (torch.Tensor): Data input for the generator.
            hidden_states (torch.Tensor): Hidden states of the generator.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Codes and hidden states. The later is referred in
                the paper as h_t (hidden states of GRU at time step t).
        """
        hidden_states = self.gru_cell(data, hidden_states)
        codes = self.hidden2codes(hidden_states)
        return codes, hidden_states

    def forward(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GRU.

        Args:
            noise (torch.Tensor): Noise for generating.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Samples and hidden states of the GRU.
        """
        codes = self.hidden2codes(noise)
        hidden_states = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []

        # TODO: check if we can make iterations till the actual number of admissions and not max
        for _ in range(self.max_number_admissions):
            samples.append(codes)
            codes, hidden_states = self.step(codes, hidden_states)
            hiddens.append(hidden_states)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens


class AttentionScore(nn.Module):
    """Compute attention scores for computing smooth conditional matrix with.

    AttentionScore is used to compute attention scores for the SmoothCondition layer.
    """

    def __init__(self, number_codes: int, attention_dim: int) -> None:
        """Constructor of the AttentionScore layer.

        Args:
            number_codes (int): Number of codes (features) in input.
            attention_dim (int): Number of visits for attention vector.
        """
        super().__init__()
        self.code_num = number_codes
        self.attention_dim = attention_dim

        # Modules for
        # - computing attention weights
        self.w_omega = nn.Linear(number_codes, attention_dim)
        # - computing attention scores for one visit
        self.u_omega = nn.Linear(attention_dim, 1)

    def forward(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the AttentionScore module.

        We compute the attention scores for the input data per visit.

        Args:
            data (torch.Tensor): Input data.
            number_admissions (torch.Tensor): Number of admissions per patient.

        Returns:
            torch.Tensor: Probabilities per visit.
        """
        t = self.w_omega(data)
        vu = self.u_omega(t).squeeze(dim=-1)

        mask = sequence_mask(number_admissions, vu.shape[-1])
        score = masked_softmax(vu, mask)
        return score


class SmoothCondition(nn.Module):
    """Smooth Condition - compute attention scores as smooth conditional matrix with AttentionScore()."""

    def __init__(self, number_codes: int, attention_dim: int) -> None:
        """Constructor. Class used to create a Smooth Condition Matrix, i.e. smoothed
        attention scores for multiple visits.

        Args:
            number_codes (int): Number of codes in the dataset.
            attention_dim (int): Attention dimension.
        """
        super().__init__()
        self.attention = AttentionScore(number_codes, attention_dim)

    def forward(
        self,
        probability_matrix: torch.Tensor,
        number_windows: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the SmoothCondition module.

        We compute the attention scores for the input data and then apply the scores to the
        target codes.

        Args:
            probability_matrix (torch.Tensor): Input data.
            number_windows (torch.Tensor): number_windows per patient.
            target_codes (torch.Tensor): Codes to generate.

        Returns:
            torch.Tensor: Smoothed data with attention scores.
        """
        score_values = self.attention(probability_matrix, number_windows)
        smoothed_attentions_scores = torch.zeros_like(probability_matrix)
        patient_indices = torch.arange(len(probability_matrix))
        smoothed_attentions_scores[patient_indices, :, target_codes] = score_values
        probability_matrix = probability_matrix + smoothed_attentions_scores

        return torch.clip(probability_matrix, max=1)

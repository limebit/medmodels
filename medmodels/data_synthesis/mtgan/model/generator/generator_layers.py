"""Generator Layers.

Equations and sections refer to the paper:
"Multi-level Generative Adversarial Networks for Synthetic Electronic Health Record Generation" by Chang Lu et al. 2023.
DOI:10.1109/TKDE.2023.3310909

This module contains the layers used in the Generator of MTGAN:
    - SyntheticGRU
    - AttentionScore
    - SmoothAttention
"""

from typing import Tuple

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import (
    apply_masked_softmax,
    find_sequence_mask,
)


class SyntheticGRU(nn.Module):
    """Synthetic GRU layer for the Generator of MTGAN."""

    hidden_dimension: int
    maximum_number_of_windows: int
    device: torch.device

    gru_cell: nn.GRUCell
    linear_layer: nn.Sequential

    def __init__(
        self,
        total_number_of_concepts: int,
        hidden_dimension: int,
        maximum_number_of_windows: int,
        device: torch.device,
    ) -> None:
        """Constructor for SyntheticGRU layer.

        Args:
            total_number_of_concepts (int): Total number of concepts in the synthetic
                data.
            hidden_dimension (int): Hidden dimension of the GRU cell.
            maximum_number_of_windows (int): Maximum number of windows for a patient.
            device (torch.device): Device where the data is located.
        """
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.maximum_number_of_windows = maximum_number_of_windows
        self.device = device
        self.to(device)

        self.gru_cell = nn.GRUCell(
            input_size=total_number_of_concepts, hidden_size=hidden_dimension
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dimension, total_number_of_concepts), nn.Sigmoid()
        )

    def step(
        self, data: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step function for the Synthetic GRU layer.

        Calls the GRU cell and the linear layer to generate the synthetic data.

        Args:
            data (torch.Tensor): Noise data tensor (shape (batch size, total number of
                concepts)).
            hidden_states (torch.Tensor): Hidden states tensor (shape (batch size,
                generator hidden dimension)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Synthetic probabilities for each concept
                in a single visit for a patient (shape batch size, total number of
                concepts), and hidden states of the synthetic GRU (shape batch size,
                generator hidden dimension).
        """
        # equation 13 from original paper (look at module docstrings on top)
        hidden_states = self.gru_cell(data, hidden_states)
        # equation 14 from original paper (look at module docstrings on top)
        synthetic_probabilities = self.linear_layer(hidden_states)

        return synthetic_probabilities, hidden_states

    def forward(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for the Synthetic GRU layer.

        Args:
            noise (torch.Tensor): Noise tensor from which to generate synthetic
                probability matrix. Shape: (batch size, generator hidden dimension).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated synthetic probabilites for
                each concept, window and patient (shape (batch size, maximum number of
                windows, total number of concepts)) and hidden states of the synthetic
                GRU (shape (batch size, generator hidden dimension)).
        """
        synthetic_probabilities = self.linear_layer(noise)
        hidden_states = torch.zeros(
            len(synthetic_probabilities), self.hidden_dimension, device=self.device
        )
        synthetic_probabilities_list = []
        hidden_states_list = []

        for _ in range(self.maximum_number_of_windows):
            synthetic_probabilities_list.append(synthetic_probabilities)
            synthetic_probabilities, hidden_states = self.step(
                synthetic_probabilities, hidden_states
            )
            hidden_states_list.append(hidden_states)

        probability_matrix = torch.stack(synthetic_probabilities_list, dim=1)
        hidden_states = torch.stack(hidden_states_list, dim=1)

        return probability_matrix, hidden_states


class AttentionScore(nn.Module):
    total_number_of_concepts: int
    attention_dimension: int
    """Computes attention scores for computing smooth conditinal matrix."""

    def __init__(self, total_number_of_concepts: int, attention_dimension: int) -> None:
        """Constructor for AttentionScore layer.

        Args:
            total_number_of_concepts (int): Total number of concepts in the synthetic
                data.
            attention_dimension (int): Dimension of the attention scores.
        """
        super().__init__()
        self.total_number_of_concepts = total_number_of_concepts
        self.attention_dimension = attention_dimension

        # Neural Networks
        # - computing attention weights
        self.attention_space_transform = nn.Linear(
            total_number_of_concepts, attention_dimension
        )
        # - computing attention scores for one visit
        self.score_transform = nn.Linear(attention_dimension, 1)

    def forward(
        self,
        probability_matrix: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the AttentionScore layer.

        It computes the attention scores for the synthetic data per visit.

        Args:
            probability_matrix (torch.Tensor): Probability matrix, i.e. synthetic data
                tensor of shape (batch size, maximum number of windows, total number of
                concepts).
            number_of_windows_per_patient (torch.Tensor): Number of windows per
                patient.

        Returns:
            torch.Tensor: Smoothed attention scores for the synthetic data per visit.
                Shape: (batch size, maximum number of windows).
        """
        attention_space = self.attention_space_transform(probability_matrix)
        attention_score_matrix = self.score_transform(attention_space).squeeze(dim=-1)

        mask = find_sequence_mask(
            number_of_windows_per_patient, attention_score_matrix.shape[-1]
        )
        smoothed_attention_score_matrix = apply_masked_softmax(
            attention_score_matrix, mask
        )

        return smoothed_attention_score_matrix


class SmoothAttention(nn.Module):
    """Soomth Attention layer for the Generator of MTGAN.

    It computes the attention scores as a smooth conditional matrix."""

    attention: AttentionScore

    def __init__(self, total_number_of_concepts: int, attention_dimension: int) -> None:
        """Constructor class used to create a Smooth Condition Matrix, i.e. smoothed
        attention scores for multiple visits.

        Args:
            total_number_of_concepts (int): Total number of concepts in the synthetic
                data.
            attention_dimension (int): Dimension of the attention scores.
        """
        super().__init__()
        self.attention = AttentionScore(total_number_of_concepts, attention_dimension)

    def forward(
        self,
        probability_matrix: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the SmoothAttention layer.

        Args:
            probability_matrix (torch.Tensor): Probability matrix, i.e. synthetic data
                tensor of shape (batch size, maximum number of windows, total number of
                concepts).
            number_of_windows_per_patient (torch.Tensor): Number of windows per
                patient.
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            torch.Tensor: The sum of the probability matrix and the smoothed attention
                scores for the synthetic data per visit. Shape: (batch size, maximum
                number of windows, total number of concepts).
        """
        smoothed_attention_score_matrix = self.attention(
            probability_matrix, number_of_windows_per_patient
        )
        smoothed_target_scores = torch.zeros_like(probability_matrix)
        smoothed_target_scores[
            torch.arange(len(probability_matrix)), :, target_concepts
        ] = smoothed_attention_score_matrix
        attention_probability_matrix = probability_matrix + smoothed_target_scores

        return torch.clip(attention_probability_matrix, max=1)

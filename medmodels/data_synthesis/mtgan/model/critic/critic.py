"""Critic for MTGAN.

This module contains the Critic class, which is used in the MTGAN model.
"""

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import sequence_mask


class Critic(nn.Module):
    """Critic for the MTGAN model."""

    def __init__(
        self,
        number_codes: int,
        critic_hidden_dimension: int,
        generator_hidden_dimension: int,
    ) -> None:
        """Constructor for Critic of MTGAN.

        When instantiated, create self.linear: A NN consisting of an exponential linear
        unit (ELU) between two Linear units.

        Args:
            number_codes (int): Size of input features (number of codes)
            critic_hidden_dimension (int): Size of hidden layers in the Critic
            generator_hidden_dimension (int): number of features in generator
        """
        super(Critic, self).__init__()

        self.number_codes = number_codes
        self.critic_hidden_dimension = critic_hidden_dimension
        self.generator_hidden_dimension = generator_hidden_dimension

        self.linear = nn.Sequential(
            nn.Linear(
                number_codes + generator_hidden_dimension, critic_hidden_dimension
            ),
            nn.ELU(inplace=True),
            nn.Linear(critic_hidden_dimension, 1),
        )

    def forward(
        self, data: torch.Tensor, hiddens: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the Critic.

        Run NN, using self.linear (defined in constructor). Uses sequence_mask for
        shaping tensor and applying mask.

        Args:
            data (torch.Tensor): input data (batch_size of patients, max_length of visits, code_num)
            hiddens (torch.Tensor): hidden state of GRU (batch_size, max_length, critic_hidden_dim)
            number_admissions (torch.Tensor): length of each patient's visit history

        Returns:
            torch.Tensor: scores of the critic. If the score is high, the data is considered to be
                real data. If the score is low, the data is considered to be synthetic data.
        """
        output = torch.cat([data, hiddens], dim=-1)
        output: torch.Tensor = self.linear(output).squeeze(dim=-1)

        max_number_admissions = data.shape[1]
        mask = sequence_mask(number_admissions, max_number_admissions)
        output *= mask
        output = output.sum(dim=-1)

        return output / number_admissions

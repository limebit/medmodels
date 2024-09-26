"""Loss function for Critic.

WGANGPLoss: Wasserstein-Loss with gradient penalty for Critic
"""

from typing import Tuple

import torch
from torch import autograd, nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic


class CriticLoss(nn.Module):
    """Critic Loss: Wasserstein-Loss with gradient penalty for Critic."""

    def __init__(self, critic: Critic, lambda_: float) -> None:
        """Constructor for the Critic Wasserstein-Loss with gradient penalty.

        Args:
            critic (Critic): Critic
            lambda_ (int): Gradient penalty coefficient.
        """
        super().__init__()
        self.critic = critic
        self.lambda_ = lambda_

    def forward(
        self,
        real_data: torch.Tensor,
        real_hiddens: torch.Tensor,
        synthetic_data: torch.Tensor,
        synthetic_hiddens: torch.Tensor,
        number_windows_per_patient: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Wasserstein-distance and return Loss with gradient penalty and Wasserstein-distance.

        Args:
            real_data (torch.Tensor): Real data of shape (batch size, maximum number
                of windows, total number of concepts)
            real_hiddens (torch.Tensor): Real hidden states from RealGRU of shape
                (batch size, maximum number of windows, generator hidden dimension)
            synthetic_data (torch.Tensor): Synthetic data generated, same shape as
                real data
            synthetic_hiddens (torch.Tensor): Synthetic hidden states from SyntheticGRU
                with the same shape as real hiddens
            number_windows_per_patient (torch.Tensor): number of windows per patient

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Critic loss and Wasserstein distance
        """
        critic_real = self.critic(real_data, real_hiddens, number_windows_per_patient)
        critic_synthetic = self.critic(
            synthetic_data, synthetic_hiddens, number_windows_per_patient
        )
        gradient_penalty = self.compute_gradient_penalty(
            real_data,
            real_hiddens,
            synthetic_data,
            synthetic_hiddens,
            number_windows_per_patient,
        )
        wasserstein_distance = critic_real.mean() - critic_synthetic.mean()
        critic_loss = -wasserstein_distance + gradient_penalty
        return critic_loss, wasserstein_distance

    def compute_gradient_penalty(
        self,
        real_data: torch.Tensor,
        real_hiddens: torch.Tensor,
        synthetic_data: torch.Tensor,
        synthetic_hiddens: torch.Tensor,
        number_windows_per_patient: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty. This is used to improve Wasserstein-GAN.

        Args:
            real_data (torch.Tensor): Real data of shape (batch size, maximum number
                of windows, total number of concepts)
            real_hiddens (torch.Tensor): Real hidden states from RealGRU of shape
                (batch size, maximum number of windows, generator hidden dimension)
            synthetic_data (torch.Tensor): Synthetic data generated, same shape as
                real data
            synthetic_hiddens (torch.Tensor): Synthetic hidden states from SyntheticGRU
                with the same shape as real hiddens
            number_windows_per_patient (torch.Tensor): number of windows per patient

        Returns:
            torch.Tensor: gradient penalty
        """
        batch_size = len(real_data)

        # Ensure no gradients are calculated for this section to save memory and compute
        with torch.no_grad():
            # Alpha for interpolation, defines the ratio between real vs synthetic data
            alpha = torch.rand((batch_size, 1, 1)).to(real_data.device)

            interpolated_data = alpha * real_data + (1 - alpha) * synthetic_data
            interpolated_hiddens = (
                alpha * real_hiddens + (1 - alpha) * synthetic_hiddens
            )

        # Set required gradient attributes for data to allow gradient computations
        interpolated_data.requires_grad_(True)
        interpolated_hiddens.requires_grad_(True)

        # Compute gradients of the critic outputs with respect to data and hidden states
        critic_outputs = self.critic(
            interpolated_data, interpolated_hiddens, number_windows_per_patient
        )
        gradients = autograd.grad(
            outputs=critic_outputs,
            inputs=[interpolated_data, interpolated_hiddens],
            grad_outputs=torch.ones_like(critic_outputs),
            create_graph=True,
            retain_graph=True,
        )
        gradients = torch.cat(gradients, dim=-1)
        gradients = gradients.view(len(gradients), -1)
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2

        # Return the scaled gradient penalty
        return gradient_penalty.mean() * self.lambda_

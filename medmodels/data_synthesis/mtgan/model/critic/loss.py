"""Loss function for Critic.

WGANGPLoss: Wasserstein-Loss with gradient penalty for Critic
"""

from typing import Tuple

import torch
from torch import autograd, nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic


class WGANGPLoss(nn.Module):
    """Wasserstein-Loss with gradient penalty for Critic."""

    def __init__(self, critic: Critic, lambda_: float = 10) -> None:
        """Constructor for the Wasserstein-Loss with gradient penalty for Critic.

        Args:
            critic (Critic): Critic
            lambda_ (int, optional): Gradient penalty coefficient. Defaults to 10.
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
        number_admissions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Wasserstein-distance and return Loss with gradient penalty and Wasserstein-distance.

        Args:
            real_data (torch.Tensor): Real data
            real_hiddens (torch.Tensor): Real hidden states of BaseGRU
            synthetic_data (torch.Tensor): Generated data
            synthetic_hiddens (torch.Tensor): Generated hidden states
            number_admissions (torch.Tensor): Number of admissions per patient
        """
        critic_real: torch.Tensor = self.critic(
            real_data, real_hiddens, number_admissions
        )
        critic_synthetic: torch.Tensor = self.critic(
            synthetic_data, synthetic_hiddens, number_admissions
        )
        gradient_penalty = self.compute_gradient_penalty(
            real_data,
            real_hiddens,
            synthetic_data,
            synthetic_hiddens,
            number_admissions,
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
        number_admissions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty. This is used to improve Wasserstein-GAN.

        Args:
            real_data (torch.Tensor): real data
            real_hiddens (torch.Tensor): real hidden states of BaseGRU
            synthetic_data (torch.Tensor): generated data
            synthetic_hiddens (torch.Tensor): generated hidden states
            number_admissions (torch.Tensor): number of visits per patient

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
            interpolated_data, interpolated_hiddens, number_admissions
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

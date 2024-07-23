"""Module for training the Critic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.critic.loss import WGANGPLoss
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
)

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.train.gan_trainer import (
        TrainingHyperparametersTotal,
    )


class CriticTrainer:
    """Class for training the critic."""

    def __init__(
        self,
        critic: Critic,
        generator: Generator,
        real_gru: RealGRU,
        hyperparameters: TrainingHyperparametersTotal,
        device: Union[torch.device, torch.cuda.device],
    ):
        """Trains the Critic using the Wasserstein GAN with Gradient Penalty loss.

        Args:
            critic (Critic): Critic instance
            generator (Generator): Generator instance
            real_gru (RealGRU): RealGRU instance
            hyperparams (TrainingHyperparametersTotal): hyperparameters dictionary
        """
        self.critic = critic
        self.generator = generator
        self.real_gru = real_gru
        self.device = device
        self.batch_size = hyperparameters.get("batch_size")
        self.iterations = hyperparameters.get("critic_iterations")

        self.optimizer = torch.optim.adam.Adam(
            critic.parameters(),
            lr=hyperparameters.get("critic_lr"),
            betas=(hyperparameters.get("beta0"), hyperparameters.get("beta1")),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=hyperparameters.get("decay_step"),
            gamma=hyperparameters.get("decay_rate"),
        )
        self.loss_fn = WGANGPLoss(
            critic, lambda_=hyperparameters.get("lambda_gradient")
        )

    def _step(
        self,
        real_data: torch.Tensor,
        number_admissions: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> Tuple[float, float]:
        """Helper function for step: computing loss and apply backpropagation with optimizer.

        Args:
            real_data (torch.Tensor): real data
            number_admissions (torch.Tensor): number of admissions per patient
            target_codes (torch.Tensor): target codes

        Returns:
            Tuple[float, float]: loss, wasserstein distance
        """
        real_hiddens = self.real_gru.calculate_hidden(real_data, number_admissions)
        synthetic_data, synthetic_hiddens = self.generator.sample_with_hidden_states(
            target_codes,
            number_admissions,
        )
        loss, wasserstein_distance = self.loss_fn(
            real_data,
            real_hiddens,
            synthetic_data,
            synthetic_hiddens,
            number_admissions,
        )
        self.optimizer.zero_grad()
        loss.backward()

        # TODO: Check if Gradient clipping is needed
        # clip_value = 1.0
        # clip_grad_norm_(self.critic.parameters(), clip_value)

        self.optimizer.step()
        return loss.item(), wasserstein_distance.item()

    def step(
        self,
        real_data: torch.Tensor,
        real_number_admissions: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> Tuple[float, float]:
        """Training step, including evaluation of the generator.

        Args:
            real_data (torch.Tensor): real data
            real_number_admissions (torch.Tensor): length of real data
            target_codes (torch.Tensor): target codes
        """
        # Generator should not be trained, but used to generate samples, so set to eval,
        # Critic should be trained, so set to train
        self.critic.train()
        self.generator.eval()

        loss, wasserstein_distance = 0, 0
        for _ in range(self.iterations):
            loss_iteration, wasserstein_distance_iteration = self._step(
                real_data, real_number_admissions, target_codes
            )
            loss += loss_iteration
            wasserstein_distance += wasserstein_distance_iteration
        loss /= self.iterations
        wasserstein_distance /= self.iterations
        self.scheduler.step()

        return loss, wasserstein_distance

    def evaluate(
        self,
        data_loader: MTGANDataLoader,
    ) -> float:
        """Evaluate the critic (on test data).

        Args:
            data_loader (MTGANDataLoader): data loader

        Returns:
            float: loss
        """
        self.critic.eval()
        with torch.no_grad():
            loss = 0
            for data in data_loader:
                data, number_admissions = data
                hidden_states = self.real_gru.calculate_hidden(data, number_admissions)
                loss += (
                    self.critic(data, hidden_states, number_admissions).mean().item()
                )
            loss = -loss / len(data_loader)
            return loss

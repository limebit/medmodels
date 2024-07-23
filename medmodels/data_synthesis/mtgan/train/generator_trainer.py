"""Generator Trainer for MTGAN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic

# from torch.nn.utils import clip_grad_norm_
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.train.gan_trainer import (
        TrainingHyperparametersTotal,
    )


class GeneratorTrainer:
    """Trainer for Generator."""

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        hyperparameters: TrainingHyperparametersTotal,
    ) -> None:
        """Constructor for Generator Trainer.

        Args:
            generator (Generator): Generator instance
            critic (Critic): Critic instance
            hyperparams (TrainingHyperparametersTotal): hyperparameters dictionary
        """
        self.generator = generator
        self.critic = critic
        self.batch_size = hyperparameters.get("batch_size")
        self.iterations = hyperparameters.get("generator_iterations")

        self.code_num = self.generator.number_codes
        self.optimizer = torch.optim.adam.Adam(
            generator.parameters(),
            lr=hyperparameters.get("generator_lr"),
            betas=(hyperparameters.get("beta0"), hyperparameters.get("beta1")),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=hyperparameters.get("decay_step"),
            gamma=hyperparameters.get("decay_rate"),
        )
        self.device = self.generator.device

    def _step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float:
        """Helper function for step: backpropagation with Adams algorithm. Compute loss
        with critic (cf. [1], equation (3) from the MTGAN paper):
            L(generator) = - |E[D(G(z))]

        Args:
            target_codes (torch.Tensor): target codes
            number_admissions (torch.Tensor): number of admissions for each patient

        Returns:
            float: loss
        """
        num_patients = len(number_admissions)

        noise = self.generator.get_noise(num_patients)
        samples, hiddens = self.generator(target_codes, number_admissions, noise)
        output: torch.Tensor = self.critic(samples, hiddens, number_admissions)
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()

        # TODO: Check if Gradient clipping is needed
        # clip_value = 1.0
        # clip_grad_norm_(self.critic.parameters(), clip_value)

        self.optimizer.step()
        return loss.item()

    def step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float:
        """Training step. Apply _step and LR-scheduler-step
        (torch.optim.lr_scheduler.StepLR).

        Args:
            target_codes (torch.Tensor): target codes
            number_admissions (torch.Tensor): number of admissions for each patient

        Returns:
            torch.Tensor: loss
        """
        self.generator.train()
        self.critic.eval()

        loss = 0
        for _ in range(self.iterations):
            loss_i = self._step(target_codes, number_admissions)
            loss += loss_i
        loss /= self.iterations

        self.scheduler.step()
        return loss

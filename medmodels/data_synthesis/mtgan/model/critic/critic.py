"""Critic for MTGAN."""

from typing import Tuple

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.loss import CriticLoss
from medmodels.data_synthesis.mtgan.model.gan import TrainingHyperparameters
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader
from medmodels.data_synthesis.mtgan.model.masks import find_sequence_mask
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU


class Critic(nn.Module):
    """Critic for MTGAN.

    The Critic tries to distinguish between real data and synthetic data generated
    by the Generator. The Critic is trained to maximize the Wasserstein distance
    between the real and synthetic data distributions.
    """

    real_gru: RealGRU
    device: torch.device

    total_number_of_concepts: int
    critic_iterations: int
    batch_size: int

    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_function: nn.Module
    critic_layers: nn.Sequential

    def __init__(
        self,
        real_gru: RealGRU,
        total_number_of_concepts: int,
        hyperparameters: TrainingHyperparameters,
        device: torch.device,
    ) -> None:
        """Constructor for Critic of MTGAN.

        The Critic tries to distinguish between real data and synthetic data generated
        by the Generator. The Critic is trained to maximize the Wasserstein distance
        between the real and synthetic data distributions.

        Args:
            real_gru (RealGRU): Real GRU model.
            total_number_of_concepts (int): Size of input features (number of codes).
            hyperparameters (TrainingHyperparameters): Hyperparameters for training.
            device (torch.device): Device to run the model on.
        """
        super().__init__()

        self.total_number_of_concepts = total_number_of_concepts
        self.real_gru = real_gru
        self.device = device
        self.to(device)

        self.batch_size = hyperparameters["batch_size"]
        self.critic_iterations = hyperparameters["critic_iterations"]

        self.optimizer = torch.optim.adam.Adam(
            self.parameters(),
            lr=hyperparameters["critic_learning_rate"],
            betas=(hyperparameters["beta0"], hyperparameters["beta1"]),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=hyperparameters["decay_step"],
            gamma=hyperparameters["decay_rate"],
        )
        self.loss_function = CriticLoss(
            self, lambda_gradient=hyperparameters["lambda_gradient"]
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(
                total_number_of_concepts
                + hyperparameters["generator_hidden_dimension"],
                hyperparameters["critic_hidden_dimension"],
            ),
            nn.ELU(inplace=True),
            nn.Linear(hyperparameters["critic_hidden_dimension"], 1),
        )

    def forward(
        self,
        data: torch.Tensor,
        hidden_states: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Critic.

        Run the critic layers, which give out an output score for each window of each
        patient. The scores are then averaged for each patient.

        Args:
            data (torch.Tensor): input data, of shape (batch size, maximum number of
                windows, total number of concepts).
            hidden_states (torch.Tensor): hidden states of the RealGRU, of shape
                (batch size, maximum number of windows, generator hidden dimension).
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).

        Returns:
            torch.Tensor: scores of the critic. If the score is high, the data is considered to be
                real data. If the score is low, the data is considered to be synthetic data.
        """
        data = torch.cat([data, hidden_states], dim=-1)
        critic_scores_per_visit = self.critic_layers(data).squeeze(dim=-1)

        maximum_number_of_windows = data.shape[1]
        mask = find_sequence_mask(
            number_of_windows_per_patient, maximum_number_of_windows
        )
        critic_scores_per_visit *= mask
        critic_scores = critic_scores_per_visit.sum(dim=-1)

        return critic_scores / number_of_windows_per_patient

    def _train_critic_iteration(
        self,
        real_data: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
        generator: Generator,
        target_concepts: torch.Tensor,
    ) -> Tuple[float, float]:
        """Train the Critic for one iteration.

        Performs the training of the Critic for one iteration, using real data and
        synthetic data generated by the Generator.

        Args:
            real_data (torch.Tensor): Real data, of shape (batch size, maximum number of
                windows, total number of concepts).
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            generator (Generator): Generator of the MTGAN, counterpart of the Critic in
                the GAN architecture.
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            Tuple[float, float]: Critic loss and Wasserstein distance between real and
                synthetic data for a single iteration.
        """
        real_hiddens = self.real_gru.calculate_hidden(
            real_data, number_of_windows_per_patient
        )
        synthetic_data, synthetic_hiddens = (
            generator._generate_samples_with_hidden_states(
                number_of_windows_per_patient, target_concepts
            )
        )
        loss, wasserstein_distance = self.loss_function(
            real_data,
            real_hiddens,
            synthetic_data,
            synthetic_hiddens,
            number_of_windows_per_patient,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), wasserstein_distance.item()

    def train_critic(
        self,
        real_data: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
        generator: Generator,
        target_concepts: torch.Tensor,
    ) -> Tuple[float, float]:
        """Train the Critic.

        Train the Critic for a number of iterations, using real data and synthetic data
        generated by the Generator.

        Args:
            real_data (torch.Tensor): Real data, of shape (batch size, maximum number of
                windows, total number of concepts).
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            generator (Generator): Generator of the MTGAN, counterpart of the Critic in
                the GAN architecture.
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            Tuple[float, float]: Critic loss and Wasserstein distance between real and
                synthetic data.
        """
        self.train()
        generator.eval()

        loss = 0
        wasserstein_distance = 0

        for _ in range(self.critic_iterations):
            loss_iteration, wasserstein_distance_iteration = (
                self._train_critic_iteration(
                    real_data, number_of_windows_per_patient, generator, target_concepts
                )
            )

            loss += loss_iteration
            wasserstein_distance += wasserstein_distance_iteration

        loss /= self.critic_iterations
        wasserstein_distance /= self.critic_iterations

        self.scheduler.step()

        return loss, wasserstein_distance

    def evaluate(self, test_data: MTGANDataLoader) -> float:
        """Evaluate the Critic on test data.

        Args:
            test_data (MTGANDataLoader): Data loader for the test data.

        Returns:
            float: Average Wasserstein distance between real and synthetic data.
        """
        self.eval()

        with torch.no_grad():
            loss = 0

            for data, number_of_windows_per_patient in test_data:
                hidden_states = self.real_gru.calculate_hidden_states(
                    data, number_of_windows_per_patient
                )

                loss += (
                    self(data, hidden_states, number_of_windows_per_patient)
                    .mean()
                    .item()
                )

            loss = -loss / len(test_data)

            return loss
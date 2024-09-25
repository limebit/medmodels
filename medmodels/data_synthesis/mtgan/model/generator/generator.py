"""Generator for MTGAN."""

from typing import Tuple

import sparse
import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparameters,
)
from medmodels.data_synthesis.mtgan.model.generator.generator_layers import (
    SmoothAttention,
    SyntheticGRU,
)
from medmodels.data_synthesis.mtgan.model.masks import find_sequence_mask


class Generator(nn.Module):
    total_number_of_concepts: int
    maximum_number_of_windows: int
    device: torch.device

    hidden_dimension: int
    attention_dimension: int
    batch_size: int
    generator_iterations: int
    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    synthetic_gru: SyntheticGRU
    smooth_attention: SmoothAttention

    def __init__(
        self,
        total_number_of_concepts: int,
        maximum_number_of_windows: int,
        hyperparameters: TrainingHyperparameters,
        device: torch.device,
    ) -> None:
        """Constructor for Generator of MTGAN.

        Args:
            total_number_of_concepts (int): Total number of concepts in the synthetic
                data.
            maximum_number_of_windows (int): Maximum number of windows for a patient.
            hyperparameters (TrainingHyperparameters): Hyperparameters for the
                Generator.
            device (torch.device): Device where the data and model are located.
        """
        super().__init__()
        self.total_number_of_concepts = total_number_of_concepts
        self.maximum_number_of_windows = maximum_number_of_windows
        self.device = device
        self.to(device)

        self.batch_size = hyperparameters["batch_size"]
        self.hidden_dimension = hyperparameters["generator_hidden_dimension"]
        self.iterations = hyperparameters["generator_iterations"]

        self.optimizer = torch.optim.adam.Adam(
            self.parameters(),
            lr=hyperparameters["generator_learning_rate"],
            betas=(hyperparameters["beta0"], hyperparameters["beta1"]),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=hyperparameters["decay_step"],
            gamma=hyperparameters["decay_rate"],
        )

        self.synthetic_gru = SyntheticGRU(
            self.total_number_of_concepts,
            self.hidden_dimension,
            self.maximum_number_of_windows,
            self.device,
        )
        self.smooth_attention = SmoothAttention(
            self.total_number_of_concepts,
            hyperparameters["generator_attention_dimension"],
        )

    def _get_target_concepts(self, batch_size: int) -> torch.Tensor:
        """Sample target concepts for each batch.

        These target concepts are drawn uniformly from all concepts to ensure all
        concepts are included in the training and synthesis process.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Target concepts.
        """
        return torch.randint(
            0, self.total_number_of_concepts, (batch_size,), device=self.device
        )

    def _generate_samples(
        self,
        number_of_windows_per_patient: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> torch.Tensor:
        """Generate synthetic data samples with respect to the target concepts.

        Args:
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            torch.Tensor: Synthetic data samples of shape (batch size, maximum number
                of windows, total number of concepts).
        """
        if isinstance(target_concepts, int):
            number_patients = 1
        else:
            number_patients = len(target_concepts)

        noise = torch.randn(number_patients, self.hidden_dimension, device=self.device)
        with torch.no_grad():
            sequence_mask = find_sequence_mask(
                number_of_windows_per_patient, self.maximum_number_of_windows
            )
            attention_probability_matrix, _ = self.forward(
                number_of_windows_per_patient, target_concepts, noise
            )
            synthetic_data = torch.bernoulli(attention_probability_matrix).to(
                attention_probability_matrix.dtype
            )
            synthetic_data *= sequence_mask

            return synthetic_data

    def _generate_samples_with_hidden_states(
        self,
        number_of_windows_per_patient: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data samples and hidden states of the synthetic GRU with respect to the target concepts.

        Args:
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Synthetic data samples of shape (batch
                size, maximum number of windows, total number of concepts) and hidden
                states of the synthetic GRU of shape (batch size, generator hidden
                dimension).
        """
        if isinstance(target_concepts, int):
            number_patients = 1
        else:
            number_patients = len(target_concepts)

        noise = torch.randn(number_patients, self.hidden_dimension, device=self.device)
        with torch.no_grad():
            sequence_mask = find_sequence_mask(
                number_of_windows_per_patient, self.maximum_number_of_windows
            )
            attention_probability_matrix, hidden_states = self.forward(
                number_of_windows_per_patient, target_concepts, noise
            )
            synthetic_data = torch.bernoulli(attention_probability_matrix).to(
                attention_probability_matrix.dtype
            )
            synthetic_data *= sequence_mask

            return synthetic_data, hidden_states

    def forward(
        self,
        number_of_windows_per_patient: torch.Tensor,
        target_concepts: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Generator.

        Args:
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.
            noise (torch.Tensor): Noise tensor from which to generate synthetic data.
                Shape: (batch size, generator hidden dimension).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated synthetic probabilites for
                each concept, window and patient (shape (batch size, maximum number of
                windows, total number of concepts)) and hidden states of the synthetic
                GRU (shape (batch size, generator hidden dimension)).
        """
        probability_matrix, hidden_states = self.synthetic_gru(noise)
        attention_probability_matrix = self.smooth_attention(
            probability_matrix, number_of_windows_per_patient, target_concepts
        )

        return attention_probability_matrix, hidden_states

    def generate_data_matrix(
        self,
        number_of_patients: int,
        windows_distribution: torch.Tensor,
        batch_size: int,
    ) -> sparse.COO:
        """Generate synthetic data matrix.

        Args:
            number_of_patients (int): Number of patients to generate data for.
            windows_distribution (torch.Tensor): Distribution of number of windows per
                patient.
            batch_size (int): Batch size for generating data.

        Returns:
            sparse.COO: Synthetic data matrix of shape (number of patients, maximum
                number of windows, total number of concepts).
        """
        synthetic_data = []
        for i in range(number_of_patients // batch_size):
            batch_number = min(batch_size, number_of_patients - i * batch_size)

            target_concepts = self._get_target_concepts(batch_number)
            number_of_windows_per_patient = torch.multinomial(
                windows_distribution, num_samples=batch_number, replacement=True
            ).to(self.device)
            synthetic_batch = (
                (self._generate_samples(number_of_windows_per_patient, target_concepts))
                .cpu()
                .numpy()
                .astype(bool)
            )
            synthetic_batch_sparse = sparse.COO.from_numpy(synthetic_batch)
            synthetic_data.append(synthetic_batch_sparse)

        return sparse.concatenate(synthetic_data, axis=0).astype(bool)

    def _train_generator_iteration(
        self,
        critic: Critic,
        number_of_windows_per_patient: torch.Tensor,
        target_concepts: torch.Tensor,
    ) -> float:
        """Training iteration for the Generator.

        Args:
            critic (Critic): Critic model
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.

        Returns:
            float: Loss of the Generator.
        """
        noise = torch.randn(
            self.batch_size,
            self.maximum_number_of_windows,
            self.hidden_dimension,
            device=self.device,
        )
        attention_probability_matrix, hidden_states = self(
            number_of_windows_per_patient, target_concepts, noise
        )
        output: torch.Tensor = critic(
            attention_probability_matrix, hidden_states, number_of_windows_per_patient
        )
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_generator(
        self,
        critic: Critic,
        target_concepts: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
    ) -> float:
        """Training step for the Generator. It can encompass multiple iterations, depending on the generator hyperparameters.

        Args:
            critic (Critic): Critic model
            target_concepts (torch.Tensor): Array of concepts chosen for each training
                batch, drawn uniformly from all concepts to ensure all concepts are
                included in the training and synthesis process. Shape: batch size.
            number_of_windows_per_patient (torch.Tensor): number of windows each
                patient has, of shape (batch size).

        Returns:
            float: Loss of the Generator.
        """
        self.train()
        critic.eval()

        loss = 0
        for _ in range(self.iterations):
            loss_iteration = self._train_generator_iteration(
                critic, target_concepts, number_of_windows_per_patient
            )
            loss += loss_iteration
        loss /= self.iterations

        self.scheduler.step()
        return loss

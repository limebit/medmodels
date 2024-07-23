"""GAN Trainer. Trains the generator and the critic at the same time."""

import json
import logging
import time
from typing import Tuple, TypedDict

import numpy as np
import torch

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataset,
)
from medmodels.data_synthesis.mtgan.model.samplers import MTGANDataSampler
from medmodels.data_synthesis.mtgan.train.critic_trainer import CriticTrainer
from medmodels.data_synthesis.mtgan.train.generator_trainer import GeneratorTrainer


class TrainingHyperparameters(TypedDict, total=False):
    batch_size: int
    real_gru_training_epochs: int
    real_gru_lr: float
    gan_training_epochs: int
    critic_hidden_dim: int
    generator_hidden_dim: int
    generator_attention_dim: int
    critic_iterations: int
    generator_iterations: int
    critic_lr: float
    generator_lr: float
    beta0: float
    beta1: float
    decay_step: int
    decay_rate: float
    lambda_gradient: float
    lambda_sparseness: float
    test_freq: int


class TrainingHyperparametersTotal(TypedDict, total=True):
    batch_size: int
    real_gru_training_epochs: int
    real_gru_lr: float
    gan_training_epochs: int
    critic_hidden_dim: int
    generator_hidden_dim: int
    generator_attention_dim: int
    critic_iterations: int
    generator_iterations: int
    critic_lr: float
    generator_lr: float
    beta0: float
    beta1: float
    decay_step: int
    decay_rate: float
    lambda_gradient: float
    lambda_sparseness: float
    test_freq: int


class GANTrainer:
    """GAN Trainer. Trains the generator and the critic at the same time."""

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        real_gru: RealGRU,
        dataset: MTGANDataset,
        hyperparameters: TrainingHyperparametersTotal,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Trains the GAN.

        Args:
            generator (Generator): Generator
            critic (Critic): Critic
            real_gru (RealGRU): RealGRU
            dataset (MTGANDataset): The dataset.
            errors_dir (Path): Path to the directory to save errors.
            hyperparameters (TrainingHyperparametersTotal): hyperparameters dictionary
            device (torch.device): device. Defaults to torch.device("cpu").
        """
        self.generator = generator
        self.critic = critic
        self.real_gru = real_gru
        self.device = device

        self.generator_trainer = GeneratorTrainer(
            generator=generator,
            critic=critic,
            hyperparameters=hyperparameters,
        )
        self.critic_trainer = CriticTrainer(
            critic=critic,
            generator=generator,
            real_gru=real_gru,
            hyperparameters=hyperparameters,
            device=device,
        )
        self.num_admissions_distribution = self._find_admissions_distribution(
            dataset=dataset
        )

        self.lambda_sparseness = hyperparameters.get("lambda_sparseness")
        self.epochs = hyperparameters.get("gan_training_epochs")
        self.batch_size = hyperparameters.get("batch_size")
        test_freq = hyperparameters.get("test_freq")
        if test_freq >= self.epochs:
            raise ValueError("test_freq hyperparameter must be smaller than epochs.")

        self.test_freq = test_freq
        self.training_sampler = MTGANDataSampler(dataset, device)

    def _find_admissions_distribution(self, dataset: MTGANDataset) -> torch.Tensor:
        """
        Find the distribution of the number of admissions per patient in the dataset.

        Args:
            dataset (MTGANDataset): The dataset.

        Returns:
            torch.Tensor: The distribution of the number of admissions per patient.
        """
        medrecord = dataset.medrecord
        number_admissions_per_patient = np.array(
            list(
                medrecord.node[
                    sorted(medrecord.nodes_in_group(dataset.patients_group)),
                    dataset.number_admissions_attribute,
                ].values()
            )
        )
        num_admissions_distribution = np.bincount(number_admissions_per_patient) / (
            len(number_admissions_per_patient)
        )
        return torch.from_numpy(num_admissions_distribution).to(self.device)

    def train(self) -> Tuple[Generator, Critic]:
        """Train the GAN and return the generator and critic."""

        epoch_counter = 0
        sparseness_loss = np.inf

        while True:
            start_time = time.time()  # Start time for the iteration
            epoch_counter += 1
            target_codes = self.generator.get_target_codes(self.batch_size)
            real_data, real_number_admissions = self.training_sampler.sample(
                target_codes
            )

            critic_loss, wasserstein_distance = self.critic_trainer.step(
                real_data=real_data,
                real_number_admissions=real_number_admissions,
                target_codes=target_codes,
            )

            generator_loss = self.generator_trainer.step(
                target_codes=target_codes,
                number_admissions=real_number_admissions,
            )
            if epoch_counter % self.test_freq == 0 or epoch_counter in (1, self.epochs):
                sparseness_loss, first_admission_loss = self.get_sparseness_loss(
                    real_data
                )
                iteration_time = time.time() - start_time

                losses = {
                    "Epoch": epoch_counter,
                    "Critic_Loss": critic_loss,
                    "Generator_Loss": generator_loss,
                    "Wasserstein_Distance": wasserstein_distance,
                    "Sparseness_Loss": sparseness_loss,
                    "First_admission_loss": first_admission_loss,
                }

                logging.info(json.dumps(losses))
                print(f"{epoch_counter} epoch: {iteration_time:.2f} seconds")

            # Convergence checks
            # TODO: min_sparseness_loss = 0.8 should be hyperparameter
            if sparseness_loss < 0.8 and epoch_counter > self.epochs:
                print(f"Converged after {epoch_counter} epochs.")
                break

            if epoch_counter >= self.epochs * 1.25:
                print("Reached maximum epochs without convergence.")
                break

        return self.generator, self.critic

    def get_sparseness_loss(
        self,
        real_data: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute the sparseness loss between real and synthetic data.

        Args:
            real_data (torch.Tensor): Real data.

        Returns:
            Tuple[float, float]: Sparseness loss and first admission loss.
        """
        synthetic_data = torch.tensor(
            self.generator.generate_(
                number_patients=real_data.shape[0],
                windows_distribution=self.num_admissions_distribution,
                batch_size=self.batch_size,
            ).todense(),
            dtype=torch.bool,
        ).to(self.device)
        sparseness_real_data = torch.sum(real_data == 1)
        sparseness_synthetic_data = torch.sum(synthetic_data == 1)

        sparseness_loss = (sparseness_synthetic_data / sparseness_real_data).to(
            self.device
        ) * self.lambda_sparseness

        first_admission_loss = synthetic_data[:, 0, :].sum() / real_data[:, 0, :].sum()

        return sparseness_loss.item(), first_admission_loss.item()

"""GAN Trainer. Trains the generator and the critic at the same time."""

from typing import Tuple, TypedDict

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataset,
)
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU
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

class GANTrainer(nn.Module):
    """GAN Trainer. Trains the generator and the critic at the same time."""

    generator: Generator
    critic: Critic
    real_gru: RealGRU
    device: torch.device

    generator_trainer: GeneratorTrainer
    critic_trainer: CriticTrainer
    number_admissions_distribution: torch.Tensor

    epochs: int
    batch_size: int
    test_frequency: int
    training_sampler: MTGANDataSampler

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        real_gru: RealGRU,
        dataset: MTGANDataset,
        hyperparameters: TrainingHyperparametersTotal,
        device: torch.device = torch.device("cpu"),
    ) -> None: ...
    def _find_admissions_distribution(self, dataset: MTGANDataset) -> torch.Tensor: ...
    def train(self) -> Tuple[Generator, Critic]: ...
    def get_sparseness_loss(
        self,
        real_data: torch.Tensor,
    ) -> Tuple[float, float]: ...

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

class TrainingHyperparameters(TypedDict, total=True):
    batch_size: int
    real_gru_training_epochs: int
    real_gru_learning_rate: float
    gan_training_epochs: int
    critic_hidden_dimension: int
    generator_hidden_dimension: int
    generator_attention_dimension: int
    critic_iterations: int
    generator_iterations: int
    critic_learning_rate: float
    generator_learning_rate: float
    beta0: float
    beta1: float
    decay_step: int
    decay_rate: float
    lambda_gradient: float
    lambda_sparseness: float
    test_frequency: int

class TrainingHyperparametersOptional(TypedDict, total=False):
    batch_size: int
    real_gru_training_epochs: int
    real_gru_learning_rate: float
    gan_training_epochs: int
    critic_hidden_dimension: int
    generator_hidden_dimension: int
    generator_attention_dimension: int
    critic_iterations: int
    generator_iterations: int
    critic_learning_rate: float
    generator_learning_rate: float
    beta0: float
    beta1: float
    decay_step: int
    decay_rate: float
    lambda_gradient: float
    lambda_sparseness: float
    test_frequency: int

class GAN(nn.Module):
    generator: Generator
    critic: Critic
    real_gru: RealGRU
    device: torch.device

    number_of_windows_distribution: torch.Tensor
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
        hyperparameters: TrainingHyperparameters,
        device: torch.device,
    ) -> None: ...
    def train(self) -> Tuple[Generator, Critic]: ...

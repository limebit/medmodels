import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.train.gan_trainer import (
    TrainingHyperparametersTotal,
)

class GeneratorTrainer(nn.Module):
    generator: Generator
    critic: Critic
    device: torch.device

    batch_size: int
    generator_iterations: int
    number_codes: int
    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        hyperparameters: TrainingHyperparametersTotal,
    ) -> None: ...
    def _step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float: ...
    def step(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> float: ...

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.gan import TrainingHyperparameters
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU

class Critic(nn.Module):
    real_gru: RealGRU
    device: torch.device

    total_number_of_concepts: int
    critic_hidden_dimension: int
    generator_hidden_dimension: int
    critic_iterations: int
    batch_size: int

    optimizer: torch.optim.optimizer.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_function: nn.Module
    linear_layer: nn.Sequential

    def __init__(
        self,
        real_gru: RealGRU,
        total_number_of_concepts: int,
        hyperparameters: TrainingHyperparameters,
        device: torch.device,
    ) -> None: ...
    def forward(
        self, data: torch.Tensor, hiddens: torch.Tensor, number_windows: torch.Tensor
    ) -> torch.Tensor: ...
    def train(
        self,
        real_data: torch.Tensor,
        real_number_windows: torch.Tensor,
        generator: Generator,
        target_concepts: torch.Tensor,
    ) -> float: ...
    def evaluate(self, data_loader: MTGANDataLoader) -> float: ...

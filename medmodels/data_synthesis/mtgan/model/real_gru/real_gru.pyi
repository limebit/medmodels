import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparameters,
)
from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader

class RealGRU(nn.Module):
    total_number_of_concepts: int
    gru: nn.GRU
    linear_layer: nn.Sequential

    epochs: int

    def __init__(
        self,
        total_number_of_concepts: int,
        hyperparamaters: TrainingHyperparameters,
    ) -> None: ...
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...
    def calculate_hidden(
        self, data: torch.Tensor, number_windows: torch.Tensor
    ) -> torch.Tensor: ...
    def train(self, train_loader: MTGANDataLoader) -> None: ...

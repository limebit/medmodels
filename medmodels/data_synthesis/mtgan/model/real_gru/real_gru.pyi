import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader

class RealGRU(nn.Module):
    number_codes: int
    gru: nn.GRU
    linear_layer: nn.Sequential

    epochs: int

    def __init__(
        self, number_codes: int, hyperparamaters: TrainingHyperparametersTotal
    ) -> None: ...
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...
    def calculate_hidden(
        self, data: torch.Tensor, number_admissions: torch.Tensor
    ) -> torch.Tensor: ...
    def train(self, train_loader: MTGANDataLoader) -> None: ...

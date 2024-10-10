"""This module defines the SynthesizerModel class, which is a PyTorch module that
is used to generate synthetic data using a synthesizer model, after the synthesizer
model has been trained on real data."""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import sparse
import torch
from torch import nn

from medmodels import MedRecord


class SynthesizerModel(nn.Module, metaclass=ABCMeta):
    """SynthesizerModel is an abstract class to define synthesizer models.

    This class should be instantiated once a Synthesizer has been trained on real data.
    """

    number_of_samples: int
    device: torch.device

    def __init__(
        self,
        number_of_samples: int,
        device: torch.device,
    ) -> None:
        """Initializes the SynthesizerModel with the required components.

        Args:
            number_of_samples (int): Number of samples to generate.
            device (torch.device): Device to use for the model.
        """
        super().__init__()
        self.number_of_samples = number_of_samples
        self.device = device
        self.to(self.device)

    @abstractmethod
    def forward(self) -> Union[torch.Tensor, sparse.COO]:
        """Generates synthetic data using the trained synthesizer model.

        Returns:
            Union[torch.Tensor, sparse.COO]: Synthetic data generated by the model.
        """
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Saves the trained synthesizer model to a file.

        Args:
            path (Path): Path to save the model to.
        """
        pass

    @abstractmethod
    def generate_synthetic_data(self) -> MedRecord:
        """Generates synthetic data using the trained synthesizer model.

        To implement this method, the child class should call the forward method and
        the postprocess method to generate synthetic data.

        Returns:
            MedRecord: Synthetic data generated by the model.
        """
        pass

    @abstractmethod
    def postprocess(self, synthetic_data: Union[MedRecord, sparse.COO]) -> MedRecord:
        """Postprocesses the synthetic data generated by the model.

        Args:
            synthetic_data (Union[MedRecord, sparse.COO]): Synthetic data generated by the model.

        Returns:
            MedRecord: Postprocessed synthetic data.
        """
        pass
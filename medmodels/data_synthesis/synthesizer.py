"""Parent class for synthesizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import torch
from torch import nn

if TYPE_CHECKING:
    from medmodels.data_synthesis.builder import SynthesizerBuilder


class Synthesizer(nn.Module):
    """Synthesizer is the parent class to define synthesizers."""

    _preprocessor: Type[nn.Module]
    _postprocessor: Type[nn.Module]
    device: torch.device

    def __init__(
        self,
        preprocessor: Type[nn.Module],
        postprocessor: Type[nn.Module],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes the synthesizer with the required components.

        Args:
            preprocessor (Type[nn.Module]): Type of preprocessor module.
            postprocessor (Type[nn.Module]): Type of postprocessor module.
            device (torch.device): Device to use for the model. Defaults to CPU.
        """
        super().__init__()
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.device = device
        self.to(self.device)

    @staticmethod
    def builder() -> SynthesizerBuilder:
        """Creates a SynthesizerBuilder instance.

        Returns:
            SynthesizerBuilder: An instance of SynthesizerBuilder.
        """
        from medmodels.data_synthesis.builder import SynthesizerBuilder

        return SynthesizerBuilder()

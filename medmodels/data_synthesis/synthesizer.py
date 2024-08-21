"""Abstract class for synthesizers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

from medmodels.data_synthesis.builder import SynthesizerBuilder
from medmodels.medrecord.medrecord import MedRecord

if TYPE_CHECKING:
    from medmodels.data_synthesis.synthesizer_model import SynthesizerModel


class Synthesizer(nn.Module):
    """Synthesizer is an abstract class that serves as a blueprint for synthesizers.

    It ensures the correct instantiation, training, and persistence of
    models.
    """

    preprocessor: nn.Module
    postprocessor: nn.Module
    device: torch.device

    def __init__(
        self,
        preprocessor: nn.Module,
        postprocessor: nn.Module,
    ) -> None: ...
    @classmethod
    def builder(cls) -> SynthesizerBuilder: ...
    def preprocess(self, medrecord: MedRecord) -> MedRecord: ...
    def fit(
        self, medrecord: MedRecord, save_directory: Optional[Path]
    ) -> SynthesizerModel: ...
    def fit_from(
        self,
        medrecord: MedRecord,
        save_directory: Optional[Path],
        **kwargs: Dict[str, Path],
    ) -> SynthesizerModel: ...
    def save_model(self, path: Path) -> None: ...
    def load_model(self, path: Path) -> SynthesizerModel: ...

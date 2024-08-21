from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from medmodels.data_synthesis.builder import SynthesizerBuilder
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel
from medmodels.medrecord.medrecord import MedRecord

class Synthesizer(nn.Module, metaclass=ABCMeta):
    _preprocessor: nn.Module
    _postprocessor: nn.Module
    device: torch.device

    def __init__(
        self,
        preprocessor: nn.Module,
        postprocessor: nn.Module,
    ) -> None: ...
    @classmethod
    def builder(cls) -> SynthesizerBuilder: ...
    def preprocess(self, medrecord: MedRecord) -> MedRecord: ...
    @abstractmethod
    def fit(
        self, medrecord: MedRecord, checkpoint_directory: Optional[Path]
    ) -> SynthesizerModel: ...
    @abstractmethod
    def fit_from(
        self,
        medrecord: MedRecord,
        checkpoint_directory: Optional[Path],
        **kwargs: Dict[str, Path],
    ) -> SynthesizerModel: ...
    @abstractmethod
    def load_model(self, path: Path) -> SynthesizerModel: ...

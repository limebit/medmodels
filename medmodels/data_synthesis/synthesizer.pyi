from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

import torch
from torch import nn

from medmodels.data_synthesis.builder import SynthesizerBuilder
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel
from medmodels.medrecord.medrecord import MedRecord

class Synthesizer(nn.Module, metaclass=ABCMeta):
    _preprocessor: Type[nn.Module]
    _postprocessor: Type[nn.Module]
    device: torch.device

    def __init__(
        self,
        preprocessor: Type[nn.Module],
        postprocessor: Type[nn.Module],
    ) -> None: ...
    @classmethod
    def builder(cls) -> SynthesizerBuilder: ...
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
    def load_model(self, medrecord: MedRecord, path: Path) -> SynthesizerModel: ...

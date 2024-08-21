from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from medmodels.data_synthesis.builder import SynthesizerBuilder
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel
from medmodels.medrecord.medrecord import MedRecord

class Synthesizer(nn.Module):
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

from __future__ import annotations

import sparse
import torch
from typing_extensions import TYPE_CHECKING

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.mtgan import MTGAN

class MTGANModel(SynthesizerModel):
    def __init__(
        self, medrecord: MedRecord, mtgan: MTGAN, generator: Generator
    ) -> None: ...
    def _find_admissions_distribution(
        self, medrecord: MedRecord, preprocessor: MTGANPreprocessor
    ) -> torch.Tensor: ...
    def forward(
        self,
        noise: torch.Tensor,
    ) -> sparse.COO: ...

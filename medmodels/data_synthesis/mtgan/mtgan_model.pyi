from pathlib import Path
from typing import Union

import sparse
import torch

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.modules.postprocessor import MTGANPostprocessor
from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel

class MTGANModel(SynthesizerModel):
    _medrecord: MedRecord
    _preprocessor: MTGANPreprocessor
    _postprocessor: MTGANPostprocessor
    _generator: Generator

    number_of_samples: int
    batch_size: int

    def __init__(
        self,
        medrecord: MedRecord,
        preprocessor: MTGANPreprocessor,
        postprocessor: MTGANPostprocessor,
        generator: Generator,
    ) -> None: ...
    def forward(
        self,
        noise: torch.Tensor,
    ) -> sparse.COO: ...
    def save_model(
        self,
        path: Path,
    ) -> None: ...
    def generate_synthetic_data(
        self,
    ) -> MedRecord: ...
    def postprocess(
        self,
        synthetic_data: Union[MedRecord, sparse.COO],
    ) -> MedRecord: ...

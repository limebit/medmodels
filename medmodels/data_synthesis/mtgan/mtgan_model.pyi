import sparse
import torch

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.modules.postprocessor import MTGANPostprocessor
from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.data_synthesis.mtgan.mtgan import MTGAN
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel

class MTGANModel(SynthesizerModel):
    medrecord: MedRecord
    preprocessor: MTGANPreprocessor
    postprocessor: MTGANPostprocessor
    generator: Generator

    number_samples: int
    batch_size: int

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

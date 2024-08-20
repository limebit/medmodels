"""This module defines the SynthesizerModel class, which is a PyTorch module that
is used to generate synthetic data using a synthesizer model, after the synthesizer
model has been trained on real data."""

from typing import Union

import sparse
import torch
from torch import nn

from medmodels import MedRecord
from medmodels.data_synthesis.synthesizer import Synthesizer

class SynthesizerModel(nn.Module):
    number_samples: int
    device: torch.device

    def __init__(
        self,
        synthesizer: Synthesizer,
    ) -> None: ...
    def forward(self, number_samples: int) -> Union[torch.Tensor, sparse.COO]: ...
    def generate_synthetic_data(self) -> MedRecord: ...
    def postprocess(
        self, synthetic_data: Union[MedRecord, sparse.COO]
    ) -> MedRecord: ...

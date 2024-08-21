from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import sparse
import torch
from torch import nn

from medmodels import MedRecord
from medmodels.data_synthesis.synthesizer import Synthesizer

class SynthesizerModel(nn.Module, metaclass=ABCMeta):
    number_samples: int
    device: torch.device

    def __init__(
        self,
        synthesizer: Synthesizer,
        number_samples: int,
        device: torch.device,
        **kwargs: Dict[str, Any],
    ) -> None: ...
    @abstractmethod
    def forward(self) -> Union[torch.Tensor, sparse.COO]: ...
    @abstractmethod
    def save_model(self, path: Path) -> None: ...
    def generate_synthetic_data(self) -> MedRecord: ...
    def postprocess(
        self, synthetic_data: Union[MedRecord, sparse.COO]
    ) -> MedRecord: ...

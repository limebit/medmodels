from typing import List, Tuple

import torch

from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataset

class CodeSampleIterator:
    code: int
    patient_indices: List[int]

    current_index: int
    length: int

    def __init__(
        self, code: int, patients_indices: List[int], shuffle: bool = True
    ) -> None: ...

class MTGANDataSampler:
    dataset: MTGANDataset
    device: torch.device
    size: int
    code_samples: List[CodeSampleIterator]

    def __init__(
        self,
        dataset: MTGANDataset,
        code_sample_map: List[CodeSampleIterator],
        device: torch.device,
    ) -> None: ...
    def sample(
        self, target_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

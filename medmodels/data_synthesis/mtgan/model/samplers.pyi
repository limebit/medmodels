from typing import List, Tuple

import torch

from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataset


class CodeSampleIter:
    """Class for iterating over the patients who have a specific code."""

    code: int
    patient_indices: List[int]

    current_index: int
    length: int

    def __init__(
        self, code: int, patients_indices: List[int], shuffle: bool = True
    ) -> None: ...
    def __next__(self) -> int: ...


class MTGANDataSampler:
    """Class for sampling batches of data from the EHR data."""

    dataset: MTGANDataset
    device: torch.device
    size: int
    code_samples: List[CodeSampleIter]

    def __init__(
        self,
        dataset: MTGANDataset,
        device: torch.device,
    ) -> None: ...
    def _get_code_sample_map(self, dataset: MTGANDataset) -> List[CodeSampleIter]: ...
    def sample(
        self, target_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

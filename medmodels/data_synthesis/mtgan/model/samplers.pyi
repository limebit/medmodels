from typing import List, Tuple

import torch

from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataset

class ConceptSampleIterator:
    concept: int
    patient_indices: List[int]

    current_index: int
    length: int

    def __init__(
        self, concept: int, patients_indices: List[int], shuffle: bool = True
    ) -> None: ...

class MTGANDataSampler:
    dataset: MTGANDataset
    device: torch.device
    size: int
    concept_samples: List[ConceptSampleIterator]

    def __init__(
        self,
        dataset: MTGANDataset,
        concept_sample_map: List[ConceptSampleIterator],
        device: torch.device,
    ) -> None: ...
    def sample(
        self, target_concepts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

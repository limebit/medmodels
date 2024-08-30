from typing import Dict, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from medmodels.data_synthesis.mtgan.modules.preprocessor import PreprocessingAttributes
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordValue

class MTGANDataset(Dataset[torch.Tensor]):
    medrecord: MedRecord
    device: torch.device

    patients_group: Group
    concepts_group: Group

    preprocessing_attributes: PreprocessingAttributes

    total_number_of_concepts: int
    maximum_number_of_windows: int
    concept_to_index_dict: Dict[MedRecordValue, int]

    def __init__(
        self,
        medrecord: MedRecord,
        patients_group: Group,
        concepts_group: Group,
        preprocessing_attributes: PreprocessingAttributes,
        index_to_concept_dict: Dict[int, MedRecordValue],
        device: torch.device,
    ) -> None: ...

class MTGANDatasetPrediction(MTGANDataset):
    def __init__(
        self,
        medrecord: MedRecord,
        patients_group: Group,
        concepts_group: Group,
        preprocessing_attributes: PreprocessingAttributes,
        index_to_concept_dict: Dict[int, MedRecordValue],
        device: torch.device,
    ) -> None: ...

class MTGANDataLoader(DataLoader[MTGANDataset]):
    dataset: Dataset[MTGANDataset]
    shuffle: bool
    batch_size: Optional[int]

    size: int
    patient_indices: NDArray[np.int16]
    number_of_batches: int
    counter: int

    def __init__(
        self,
        dataset: MTGANDataset,
        shuffle: bool,
        batch_size: Optional[int] = None,
    ) -> None: ...

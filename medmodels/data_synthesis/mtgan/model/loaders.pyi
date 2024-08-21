from typing import Dict, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute, MedRecordValue

class MTGANDataset(Dataset[torch.Tensor]):
    medrecord: MedRecord
    device: torch.device

    patients_group: Group
    concepts_group: Group

    time_window_attribute: MedRecordAttribute
    absolute_time_window_attribute: MedRecordAttribute
    concept_index_attribute: MedRecordAttribute
    concept_edge_attribute: MedRecordAttribute
    number_admissions_attribute: MedRecordAttribute

    number_codes: int
    max_number_admissions: int
    concept_to_index_dict: Dict[MedRecordValue, int]

    def __init__(
        self,
        medrecord: MedRecord,
        preprocesor: MTGANPreprocessor,
        device: torch.device,
    ) -> None: ...

class MTGANDatasetPrediction(MTGANDataset):
    def __init__(
        self,
        medrecord: MedRecord,
        preprocessor: MTGANPreprocessor,
        device: torch.device,
    ) -> None: ...

class MTGANDataLoader(DataLoader[MTGANDataset]):
    dataset: Dataset[MTGANDataset]
    shuffle: bool
    batch_size: Optional[int]

    size: int
    patient_indices: NDArray[np.int16]
    number_batches: int
    counter: int

    def __init__(
        self,
        dataset: MTGANDataset,
        shuffle: bool,
        batch_size: Optional[int] = None,
    ) -> None: ...

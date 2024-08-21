from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute, MedRecordValue


class MTGANDataset(Dataset[torch.Tensor]):
    """Dataset for the MTGAN model to load data from the MedRecord in sparse format."""

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
        device: torch.device = torch.device("cpu"),
    ) -> None: ...
    def __len__(self) -> int: ...
    def _convert_idx(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> List[int]: ...
    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def _transform_into_coords(
        self, patient_idx: int, array_idx: int
    ) -> NDArray[np.int16]: ...
    def get_attributes(
        self,
        idx: Union[List[int], int, torch.Tensor, NDArray[np.int16]],
        attribute: MedRecordAttribute,
    ) -> Tuple[NDArray[np.bool], NDArray[Any]]: ...


class MTGANDatasetPrediction(MTGANDataset):
    """Generates the dataset for the prediction task in the GRU."""

    def __init__(
        self,
        medrecord: MedRecord,
        preprocessor: MTGANPreprocessor,
        device: torch.device = torch.device("cpu"),
    ) -> None: ...
    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def _transform_into_coords(
        self, patient_idx: int, array_idx: int, remove_window: Literal["first", "last"]
    ) -> NDArray[np.int16]: ...


class MTGANDataLoader(DataLoader[MTGANDataset]):
    """MTGAN DataLoader for the MTGAN model."""

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
    def _get_item(self, index: int) -> torch.Tensor: ...
    def __next__(self) -> torch.Tensor: ...
    def __len__(self) -> int: ...

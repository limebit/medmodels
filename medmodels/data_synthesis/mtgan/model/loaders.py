from typing import Any, List, Literal, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import MedRecordAttribute


class MTGANDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        preprocesor: MTGANPreprocessor,
        medrecord: MedRecord,
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
    def __init__(
        self,
        preprocessor: MTGANPreprocessor,
        medrecord: MedRecord,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Generates the dataset for the prediction task in the GRU."""
        super().__init__(preprocessor, medrecord, device)

    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def _transform_into_coords(
        self, patient_idx: int, array_idx: int, remove_window: Literal["first", "last"]
    ) -> NDArray[np.int16]: ...


class MTGANDataLoader(DataLoader[torch.Tensor]):
    """MTGAN DataLoader for the MTGAN model."""

    def __init__(
        self, dataset: MTGANDataset, batch_size: int, shuffle: bool
    ) -> None: ...

    def _get_item(self, index: int) -> torch.Tensor: ...

    def __next__(self) -> torch.Tensor: ...

    def __len__(self) -> int: ...

    def __iter__(self): ...

from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict, Union

import sparse
import torch
from torch import nn
from typing_extensions import TypeAlias

from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute

AttributeType: TypeAlias = Literal["categorical", "regression", "temporal"]


class PostprocessingHyperparameters(TypedDict, total=False):
    number_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_layers: int
    batch_size: int
    number_previous_admissions: int
    top_k_codes: int


class PostprocessingHyperparametersTotal(TypedDict, total=True):
    number_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_layers: int
    batch_size: int
    number_previous_admissions: int
    top_k_codes: int


class MTGANPostprocessor(nn.Module):
    """Postprocessing class for the MTGAN model."""

    device: torch.device
    preprocessor: MTGANPreprocessor
    real_medrecord: MedRecord

    attributes_types: Dict[MedRecordAttribute, AttributeType]
    attributes_concepts_types: Dict[MedRecordAttribute, AttributeType]
    hyperparameters: PostprocessingHyperparametersTotal

    def __init__(
        self,
    ) -> None: ...
    def _load_processed_data(
        self,
        real_medrecord: MedRecord,
        preprocessor: MTGANPreprocessor,
    ) -> None: ...
    def postprocess(self, synthetic_data: sparse.COO) -> MedRecord: ...
    def _patients_categorical_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Any]]: ...
    def _patients_regression_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Union[float, int]]]: ...
    def _patients_temporal_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[datetime]]: ...
    def _find_time(
        self, synthetic_data: sparse.COO, first_admissions: List[datetime]
    ) -> Dict[MedRecordAttribute, List[datetime]]: ...
    def _concepts_categorical_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Any]]: ...
    def _concepts_regression_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Union[float, int]]]: ...
    def _concepts_temporal_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[datetime]]: ...
    def convert_to_medrecord(
        self,
        synthetic_data: sparse.COO,
        synthetic_attributes_patients: Dict[MedRecordAttribute, List[Any]],
        synthetic_attributes_concepts: Dict[MedRecordAttribute, List[Any]],
    ) -> MedRecord: ...

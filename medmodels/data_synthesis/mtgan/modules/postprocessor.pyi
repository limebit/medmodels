from typing import Any, Dict, List, Literal, TypedDict

import sparse
import torch
from torch import nn
from typing_extensions import TypeAlias

from medmodels.data_synthesis.mtgan.modules.preprocessor import PreprocessingAttributes
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import MedRecordAttribute, NodeIndex

AttributeType: TypeAlias = Literal["categorical", "continuous", "temporal"]

class PostprocessingHyperparameters(TypedDict, total=True):
    number_of_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_of_layers: int
    batch_size: int
    number_of_previous_windows: int
    top_k_concepts: int

class PostprocessingHyperparametersOptional(TypedDict, total=False):
    number_of_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_of_layers: int
    batch_size: int
    number_of_previous_windows: int
    top_k_concepts: int

class MTGANPostprocessor(nn.Module):
    medrecord: MedRecord
    preprocessing_attributes: PreprocessingAttributes
    device: torch.device

    attributes_types_patients: Dict[MedRecordAttribute, AttributeType]
    attributes_types_concepts: Dict[MedRecordAttribute, AttributeType]
    hyperparameters: PostprocessingHyperparameters

    def __init__(
        self,
        medrecord: MedRecord,
        preprocessing_attributes: PreprocessingAttributes,
        index_to_concept_dict: Dict[int, NodeIndex],
        attributes_types_patients: Dict[MedRecordAttribute, AttributeType],
        attributes_types_concepts: Dict[MedRecordAttribute, AttributeType],
        hyperparameters: PostprocessingHyperparameters,
        device: torch.device,
    ) -> None: ...
    def postprocess(self, synthetic_data: sparse.COO) -> MedRecord: ...
    def convert_to_medrecord(
        self,
        synthetic_data: sparse.COO,
        synthetic_attributes_patients: Dict[MedRecordAttribute, List[Any]],
        synthetic_attributes_concepts: Dict[MedRecordAttribute, List[Any]],
    ) -> MedRecord: ...

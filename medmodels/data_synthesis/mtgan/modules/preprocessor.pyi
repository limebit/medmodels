from typing import Dict, Tuple, TypedDict

from torch import nn

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    NodeIndex,
)

class PreprocessingHyperparameters(TypedDict, total=True):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_concepts_per_window: int
    number_sampled_patients: int

class PreprocessingHyperparametersOptional(TypedDict, total=False):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_concepts_per_window: int
    number_sampled_patients: int

class PreprocessingAttributes(TypedDict):
    first_admission_attribute: MedRecordAttribute
    time_window_attribute: MedRecordAttribute
    concept_index_attribute: MedRecordAttribute
    concept_edge_attribute: MedRecordAttribute
    number_windows_attribute: MedRecordAttribute
    absolute_time_window_attribute: MedRecordAttribute

class MTGANPreprocessor(nn.Module):
    patients_group: Group
    concepts_group: Group
    time_attribute: MedRecordAttribute

    hyperparameters: PreprocessingHyperparameters

    def __init__(
        self,
        patients_group: Group,
        concepts_group: Group,
        time_attribute: MedRecordAttribute,
        hyperparameters: PreprocessingHyperparameters,
    ) -> None: ...
    def preprocess(
        self, medrecord: MedRecord
    ) -> Tuple[MedRecord, Dict[int, NodeIndex], PreprocessingAttributes]: ...

from typing import Dict, TypedDict

from torch import nn

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    NodeIndex,
)

class PreprocessingHyperparameters(TypedDict, total=False):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_codes_per_window: int
    number_sampled_patients: int

class PreprocessingHyperparametersTotal(TypedDict, total=True):
    minimum_occurrences_concept: int
    time_interval_days: int
    minimum_codes_per_window: int
    number_sampled_patients: int

class MTGANPreprocessor(nn.Module):
    patients_group: Group
    concepts_group: Group

    time_attribute: MedRecordAttribute
    first_admission_attribute: MedRecordAttribute
    time_window_attribute: MedRecordAttribute
    concept_index_attribute: MedRecordAttribute
    concept_edge_attribute: MedRecordAttribute
    number_admissions_attribute: MedRecordAttribute
    absolute_time_window_attribute: MedRecordAttribute

    index_to_concept_dict: Dict[int, NodeIndex]
    hyperparameters: PreprocessingHyperparametersTotal

    def __init__(
        self,
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: str = "time",
    ) -> None: ...
    def preprocess(self, medrecord: MedRecord) -> MedRecord: ...

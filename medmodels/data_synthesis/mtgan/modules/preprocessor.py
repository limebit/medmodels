"""Preprocessing class for the MTGAN Model."""

from typing import Dict, List, Tuple, TypedDict

from torch import nn

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import (
    Group,
    MedRecordAttribute,
    MedRecordValue,
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
    """Preprocessing class for the MTGAN model."""

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
    def _remove_uncommon_concepts(
        self,
        medrecord: MedRecord,
        min_number_occurrences: int,
        return_dict: bool = False,
    ) -> Tuple[Dict[int, NodeIndex], MedRecord]: ...
    def _find_first_admission(self, medrecord: MedRecord) -> MedRecord: ...
    def _sample_patients(
        self, medrecord: MedRecord, num_sampled_patients: int
    ) -> MedRecord: ...
    def _get_attribute(
        self, medrecord: MedRecord, attribute_name: MedRecordAttribute
    ) -> MedRecordAttribute: ...
    def _remove_low_number_time_windows(
        self,
        medrecord: MedRecord,
        patient_index: NodeIndex,
        absolute_time_attribute: MedRecordAttribute,
        min_codes_per_window: int,
    ) -> List[MedRecordValue]: ...
    def _find_relative_times(
        self, medrecord: MedRecord, time_interval_days: int, min_codes_per_window: int
    ) -> MedRecord: ...
    def preprocess(self, medrecord: MedRecord) -> MedRecord: ...

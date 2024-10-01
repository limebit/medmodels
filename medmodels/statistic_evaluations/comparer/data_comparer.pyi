from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import polars as pl

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import (
    Group,
    GroupInputList,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    NodeIndex,
)

class DataComparer:
    @staticmethod
    def compare_cohorts(
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
    ) -> CohortSummary: ...
    @staticmethod
    def compare_cohort_attributes(
        medrecords: List[Tuple[MedRecord, DataComparerConfig]],
        attributes: MedRecordAttributeInputList,
    ) -> Dict[MedRecordAttribute, pl.DataFrame]: ...
    @staticmethod
    def calculate_absolute_relative_difference(
        control_cohort: Tuple[MedRecord, DataComparerConfig],
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
        attributes: MedRecordAttributeInputList,
    ) -> Tuple[float, pl.DataFrame]: ...
    @staticmethod
    def test_difference_attributes(
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
        attributes: MedRecordAttributeInputList,
        significance_level: float,
    ) -> Dict[MedRecordAttribute, Dict[str, Union[str, bool, float]]]: ...
    @staticmethod
    def get_top_k_concepts(
        cohort: Tuple[MedRecord, DataComparerConfig],
        top_k: int,
    ) -> Dict[Union[Group, str], List[Any]]: ...
    @staticmethod
    def get_concept_counts(
        cohort: Tuple[MedRecord, DataComparerConfig],
    ) -> List[Tuple[NodeIndex, int]]: ...
    @staticmethod
    def test_difference_top_k_concepts(
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
        top_k: int,
        significance_level: float,
    ) -> Dict[Group, Dict[str, Union[str, bool, float]]]: ...
    @staticmethod
    def calculate_distance_concepts(
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
    ) -> Dict[str, float]: ...
    @staticmethod
    def full_comparison(
        cohorts: List[Tuple[MedRecord, DataComparerConfig]],
        top_k: int,
        significance_level: float,
    ) -> Dict[Union[MedRecordAttribute, Group], Dict[str, Any]]: ...

class DataComparerConfig(TypedDict):
    patients_group: Union[Group, NodeOperation]
    time_attribute: MedRecordAttribute
    concepts_groups: GroupInputList

class CohortSummary(TypedDict):
    nodes: int
    edges: Dict[Group, int]
    attributes: MedRecordAttribute

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Union

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import (
    AttributeInfo,
    AttributeSummary,
    Group,
    GroupInputList,
    MedRecordAttribute,
    MedRecordAttributeInputList,
    NodeIndex,
)

class Cohort:
    """Configuration for a cohort for evaluation and comparison.

    Needs a medrecord and the corresponding patient group that should form the cohort. The cohort group can be a predefined group in the MedRecord or a node query.

    """

    medrecord: MedRecord
    name: str
    cohort_group: Group
    time_attribute: MedRecordAttribute
    concepts_groups: GroupInputList

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        concepts_groups: GroupInputList,
        cohort_group: Union[Group, NodeOperation] = "patients",
        time_attribute: MedRecordAttribute = "time",
    ) -> None: ...

class DataComparer:
    @staticmethod
    def compare_cohorts(
        cohorts: List[Cohort],
    ) -> Dict[Group, CohortSummary]: ...
    @staticmethod
    def compare_cohort_attributes(
        medrecords: List[Cohort],
        attributes: MedRecordAttributeInputList,
    ) -> Dict[
        Group,
        Dict[MedRecordAttribute, AttributeInfo],
    ]: ...
    @staticmethod
    def calculate_absolute_relative_difference(
        control_cohort: Cohort,
        cohorts: List[Cohort],
        attributes: MedRecordAttributeInputList,
    ) -> Tuple[float, Dict[MedRecordAttribute, float]]: ...
    @staticmethod
    def test_difference_attributes(
        cohorts: List[Cohort],
        attributes: MedRecordAttributeInputList,
        significance_level: float,
    ) -> Dict[MedRecordAttribute, TestSummary]: ...
    @staticmethod
    def get_top_k_concepts(
        cohort: Cohort,
        top_k: int,
    ) -> List[NodeIndex]: ...
    @staticmethod
    def get_concept_counts(
        cohort: Cohort,
    ) -> List[Tuple[NodeIndex, int]]: ...
    @staticmethod
    def test_difference_top_k_concepts(
        cohorts: List[Cohort],
        top_k: int,
        significance_level: float,
    ) -> Dict[Group, TestSummary]: ...
    @staticmethod
    def calculate_distance_concepts(
        cohorts: List[Cohort],
    ) -> Dict[Group, DistanceSummary]: ...
    @staticmethod
    def full_comparison(
        cohorts: List[Cohort],
        top_k: int,
        significance_level: float,
    ) -> Tuple[Dict[str, CohortSummary], ComparerSummary]: ...

class CohortSummary(TypedDict):
    """Dictionary for the cohort summary."""

    attribute_info: Dict[Group, AttributeSummary]
    top_concepts: Dict[Group, List[NodeIndex]]

class DistanceSummary(TypedDict):
    """Dictonary for the Jensen-Shannon-Divergence and normalized distance between distributions."""

    js_divergence: float
    distance: float

class ComparerSummary(TypedDict):
    """Dictionary for the comparing results."""

    attribute_tests: Dict[MedRecordAttribute, TestSummary]
    concepts_tests: Dict[Group, TestSummary]
    concepts_distance: Dict[Group, DistanceSummary]

class TestSummary(TypedDict):
    test: str
    Hypothesis: str
    not_reject: bool
    p_value: float

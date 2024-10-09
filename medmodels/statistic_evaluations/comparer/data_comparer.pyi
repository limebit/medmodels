from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict, Union

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import (
    AttributeInfo,
    AttributeSummary,
    Group,
    GroupInputList,
    MedRecordAttribute,
    NodeIndex,
)

class Cohort:
    """Configuration for a cohort for evaluation and comparison.

    Needs a medrecord and the corresponding patient group that should form the cohort.
    The cohort group can be a predefined group in the MedRecord or a node query.

    """

    medrecord: MedRecord
    name: str
    cohort_group: Group
    time_attribute: MedRecordAttribute
    attributes: Optional[Dict[str, MedRecordAttribute]]
    concepts_groups: Optional[GroupInputList]

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        cohort_group: Union[Group, NodeOperation] = "patients",
        time_attribute: MedRecordAttribute = "time",
        attributes: Optional[Dict[str, MedRecordAttribute]] = None,
        concepts_groups: Optional[GroupInputList] = None,
    ) -> None: ...

class CohortSummary(TypedDict):
    """Dictionary for the cohort summary."""

    attribute_info: Dict[Group, AttributeSummary]
    top_k_concepts: Dict[Group, List[NodeIndex]]

class DistanceSummary(TypedDict):
    """Dictonary for the Jensen-Shannon-Divergence and normalized distance between
    distributions."""

    js_divergence: float
    distance: float

class ComparerSummary(TypedDict):
    """Dictionary for the comparing results."""

    attribute_tests: Dict[MedRecordAttribute, List[TestSummary]]
    concepts_tests: Dict[Group, List[TestSummary]]
    concepts_distance: Dict[Group, DistanceSummary]

class TestSummary(TypedDict):
    test: str
    Hypothesis: str
    not_reject: bool
    p_value: float

class DataComparer:
    @staticmethod
    def compare_cohort_attribute(
        cohorts_attribute: List[Tuple[Cohort, MedRecordAttribute]],
    ) -> Dict[str, AttributeInfo]: ...
    @staticmethod
    def test_difference_attribute(
        cohorts_attribute: List[Tuple[Cohort, MedRecordAttribute]],
        significance_level: float,
    ) -> List[TestSummary]: ...
    @staticmethod
    def compare_cohorts(
        cohorts: List[Cohort],
    ) -> Dict[str, CohortSummary]: ...
    @staticmethod
    def test_difference_cohort_attributes(
        cohorts: List[Cohort],
        significance_level: float,
    ) -> Dict[str, List[TestSummary]]: ...
    @staticmethod
    def calculate_absolute_relative_difference(
        control_group: Cohort,
        case_groups: List[Cohort],
    ) -> Tuple[float, Dict[MedRecordAttribute, float]]: ...
    @staticmethod
    def get_concept_counts(
        cohort: Cohort,
    ) -> List[Tuple[NodeIndex, int]]: ...
    @staticmethod
    def get_top_k_concepts(
        cohort: Cohort,
        top_k: int,
    ) -> List[NodeIndex]: ...
    @staticmethod
    def test_difference_top_k_concepts(
        cohorts: List[Cohort],
        top_k: int,
        significance_level: float,
    ) -> Dict[Group, List[TestSummary]]: ...
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

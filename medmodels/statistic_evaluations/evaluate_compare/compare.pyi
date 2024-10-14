from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict

from medmodels.medrecord.types import (
    AttributeInfo,
    AttributeSummary,
    Group,
    MedRecordAttribute,
    NodeIndex,
)
from medmodels.statistic_evaluations.evaluate_compare.evaluate import CohortEvaluator

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

class CohortComparer:
    @staticmethod
    def compare_cohort_attribute(
        cohorts: List[CohortEvaluator],
        attribute: MedRecordAttribute,
    ) -> Dict[str, AttributeInfo]: ...
    @staticmethod
    def test_difference_attribute(
        cohorts_attribute: List[CohortEvaluator],
        attribute: MedRecordAttribute,
        significance_level: float,
    ) -> List[TestSummary]: ...
    @staticmethod
    def compare_cohorts(
        cohorts: List[CohortEvaluator],
    ) -> Dict[str, CohortSummary]: ...
    @staticmethod
    def test_difference_cohort_attributes(
        cohorts: List[CohortEvaluator],
        significance_level: float,
    ) -> Dict[str, List[TestSummary]]: ...
    @staticmethod
    def calculate_absolute_relative_difference(
        control_group: CohortEvaluator,
        case_group: CohortEvaluator,
    ) -> Tuple[float, Dict[MedRecordAttribute, float]]: ...
    @staticmethod
    def test_difference_top_k_concepts(
        cohorts: List[CohortEvaluator],
        top_k: int,
        significance_level: float,
    ) -> Dict[Group, List[TestSummary]]: ...
    @staticmethod
    def calculate_distance_concepts(
        cohorts: List[CohortEvaluator],
    ) -> Dict[Group, DistanceSummary]: ...
    @staticmethod
    def full_comparison(
        cohorts: List[CohortEvaluator],
        top_k: int,
        significance_level: float,
    ) -> Tuple[Dict[str, CohortSummary], ComparerSummary]: ...

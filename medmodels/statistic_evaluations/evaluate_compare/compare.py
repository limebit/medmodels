# ruff: noqa: D100, D103, T201
from typing import Dict, List, Optional, Tuple, TypedDict

from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    AttributeSummary,
    Group,
    MedRecordAttribute,
    NodeIndex,
)
from medmodels.statistic_evaluations.evaluate_compare.evaluate import CohortEvaluator
from medmodels.statistic_evaluations.statistical_analysis.attribute_analysis import (
    AttributeStatistics,
    extract_attribute_values,
)
from medmodels.statistic_evaluations.statistical_analysis.inferential_statistics import (
    TestSummary,
    calculate_relative_difference,
    decide_hypothesis_test,
)


class CohortSummary(TypedDict):
    attribute_info: Dict[Group, AttributeSummary]
    top_k_concepts: Dict[Group, List[NodeIndex]]


class DistanceSummary(TypedDict):
    js_divergence: float
    distance: float


class ComparerSummary(TypedDict):
    attribute_tests: Dict[MedRecordAttribute, List[TestSummary]]
    concepts_tests: Dict[Group, List[TestSummary]]
    concepts_distance: Dict[Group, DistanceSummary]


class CohortComparer:
    @staticmethod
    def compare_cohort_attribute(
        cohorts: List[CohortEvaluator],
        attribute: MedRecordAttribute,
    ) -> Dict[str, AttributeStatistics]:
        """Compare the descriptive statistics of an attribute between cohorts.

        Args:
            cohorts (List[CohortEvaluator]): _description_
            attribute (MedRecordAttribute): _description_

        Raises:
            ValueError: _description_

        Returns:
            Dict[str, AttributeStatistics]: _description_
        """
        # :TODO @Laura add representation in table
        compare_attribute = {}

        for cohort in cohorts:
            if attribute not in cohort.attribute_summary:
                msg = f"Attribute {attribute} not found in cohort {cohort.name}."
                raise ValueError(msg)

            compare_attribute[cohort.name] = cohort.attribute_summary[attribute]

        return compare_attribute

    @staticmethod
    def test_difference_attribute(
        cohorts: List[CohortEvaluator],
        attribute: MedRecordAttribute,
        significance_level: float = 0.05,
    ) -> Optional[TestSummary]:
        """Use hypothesis test to test for difference in mean or distribution.

        Args:
            cohorts (List[CohortEvaluator]): _description_
            attribute (MedRecordAttribute): _description_
            significance_level (float, optional): _description_. Defaults to 0.05.

        Raises:
            ValueError: If attribute not found in all cohorts.
            ValueError: If attribute has different type in different cohorts.

        Returns:
            Optional[TestSummary]: TestSummary if possible.
        """
        if not all(attribute in cohort.attribute_summary for cohort in cohorts):
            msg = f"Attribute {attribute} not found in all cohorts."
            raise ValueError(msg)

        samples = [
            extract_attribute_values(
                medrecord=cohort.medrecord,
                group=cohort.patient_group,
                attributes=[attribute],
                nodes_or_edges="nodes",
            )[attribute]
            for cohort in cohorts
        ]

        types = [cohort.attribute_summary[attribute]["type"] for cohort in cohorts]
        if not all(types[0] == attr_type for attr_type in types):
            msg = f"""Not all attribute types are the same, found types
                {", ".join(types)} for attribute {attribute}."""
            raise ValueError(msg)

        attr_type = AttributeType[types[0]] if types[0] != "Unstructured" else None

        if attr_type:
            return decide_hypothesis_test(
                samples,
                alpha=significance_level,
                attribute_type=attr_type,
            )

        return None

    @staticmethod
    def compare_cohorts(
        cohorts: List[CohortEvaluator],
        top_k: int = 10,
    ) -> Dict[str, CohortSummary]:
        """Get a dictionary comparing different cohorts.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts to compare.
            top_k (int): Number of top concepts to show.

        Returns:
            Dict[str, CohortSummary]: Cohorts and their summaries.
        """
        cohort_comparison = {}
        for cohort in cohorts:
            cohort_comparison[cohort.name] = {
                "attribute_info": cohort.medrecord._describe_group_nodes(
                    groups=[cohort.patient_group]
                ),
                "top_k_concepts": cohort.get_top_k_concepts(top_k=top_k),
            }
        return cohort_comparison

    @staticmethod
    def test_difference_cohort_attributes(
        cohorts: List[CohortEvaluator],
        significance_level: float,
    ) -> Dict[MedRecordAttribute, TestSummary]:
        """Test all possible cohort attributes for differences.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts.
            significance_level (float): Significance level for the test.

        Returns:
            Dict[MedRecordAttribute, TestSummary]: Name of attribute and their test
                summary.
        """
        attributes = get_intersection_attributes(cohorts=cohorts)

        test_results = {}

        for attribute in attributes:
            test_results[attribute] = CohortComparer.test_difference_attribute(
                cohorts=cohorts,
                attribute=attribute,
                significance_level=significance_level,
            )

        return test_results

    @staticmethod
    def calculate_absolute_relative_difference(
        control_group: CohortEvaluator,
        case_group: CohortEvaluator,
        attributes: Optional[List[MedRecordAttribute]] = None,
    ) -> Tuple[float, Dict[MedRecordAttribute, float]]:
        """Calculates the absolute relative mean difference as a pooled mean and for every feature.

        Calculates the absolute relative mean difference for each feature between
        control and treated sets, expressed as a percentage of the control set's mean.
        This measure provides an understanding of how much each feature's average value
        changes from the control to the treated group relative to the control.

        Args:
            control_group (CohortEvaluator): _description_
            case_group (CohortEvaluator): _description_
            attributes (Optional[List[MedRecordAttribute]], optional): _description_.
                Defaults to None.

        Returns:
            Tuple[float, Dict[MedRecordAttribute, float]]: Average difference and
                Dictionary with absolute relative difference for all available features.
        """  # noqa: W505
        if not attributes:
            attributes = get_intersection_attributes(
                cohorts=[control_group, case_group]
            )

        diff_features = {}

        for attribute in attributes:
            control_stats = control_group.attribute_summary[attribute]
            case_stats = case_group.attribute_summary[attribute]

            if (
                control_stats["type"] == "Continuous"
                and case_stats["type"] == "Continuous"
            ):
                # based on Continuous TypedDict this will never fail
                assert "mean" in control_stats
                assert "mean" in case_stats

                diff_features[attribute] = calculate_relative_difference(
                    control_stats["mean"],
                    case_stats["mean"],
                )

        average_difference = sum(diff_features.values()) / len(diff_features.values())

        return (average_difference, diff_features)

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


def get_intersection_attributes(
    cohorts: List[CohortEvaluator],
) -> List[MedRecordAttribute]:
    """Get the attributes that all cohorts have in common.

    Args:
        cohorts (List[CohortEvaluator]): List of different cohorts or groups.

    Raises:
        ValueError: If no common attributes are found.

    Returns:
        List[MedRecordAttribute]: Attributes that appear in all cohorts.
    """
    attributes = set(cohorts[0].attributes.keys())

    for i in range(1, len(cohorts) - 1):
        attributes = attributes.intersection(set(cohorts[i].attributes.keys()))

    if len(attributes) == 0:
        msg = "No common attribute found between the cohorts."
        raise ValueError(msg)

    return list(attributes)

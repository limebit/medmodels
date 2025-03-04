# ruff: noqa: D100, D103, T201
from typing import Dict, List, Optional, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    """Summary of all patient and concept attributes and their concept counts."""

    attribute_info: Dict[Group, AttributeSummary]
    concept_attribute_info: Dict[Group, AttributeSummary]
    top_k_concepts: List[Tuple[NodeIndex, int]]


class DistanceSummary(TypedDict):
    """Result for the distance calculations."""

    js_divergence: float
    distance: float


class ComparerSummary(TypedDict):
    """Summary of the comparer results."""

    attribute_tests: Dict[MedRecordAttribute, List[TestSummary]]
    concepts_tests: Dict[Group, List[TestSummary]]
    concepts_distance: Dict[Group, DistanceSummary]


class CohortComparer:
    """Comparer class containing functions to compare cohorts.

    Cohorts need to be inititalized with the CohortEvaluator class to be able to use
    these functions. They can be sub groups of patients of the same MedRecord, i.e.
    male and female patients or case and control group, or they can be patients of
    different MedRecords. In case of different MedRecords, the concepts and attributes
    can be mapped to the same names in the CohortEvaluator initialization.
    """

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
            if attribute not in cohort.attribute_summary[cohort.patient_group]:
                msg = f"Attribute {attribute} not found in cohort {cohort.name}."
                raise ValueError(msg)

            compare_attribute[cohort.name] = cohort.attribute_summary[
                cohort.patient_group
            ][attribute]

        return compare_attribute

    @staticmethod
    def compare_cohort_concept_attribute(
        cohorts: List[CohortEvaluator],
        attribute: MedRecordAttribute,
        concept: Group,
    ) -> Dict[str, AttributeStatistics]:
        """Compare the descriptive statistics of an concept attribute between cohorts.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts to compare.
            attribute (MedRecordAttribute): Attribute for edges between patients and
                concept.
            concept (Group): Concept name that has the attribute on the edge with
                patients.

        Raises:
            ValueError: If concept not found in all cohorts.
            ValueError: If concept attribute not found for all cohorts.

        Returns:
            Dict[str, AttributeStatistics]: _description_
        """
        compare_attribute = {}

        for cohort in cohorts:
            if not all(concept in cohort.attribute_summary for cohort in cohorts):
                msg = f"Concept {concept} not found in all cohort attribute summaries."
                raise ValueError(msg)

            if attribute not in cohort.attribute_summary[cohort.patient_group]:
                msg = f"Attribute {attribute} not found in cohort {cohort.name}."
                raise ValueError(msg)

            compare_attribute[cohort.name] = cohort.attribute_summary[concept][
                attribute
            ]

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
        if not all(
            attribute in cohort.attribute_summary[cohort.patient_group]
            for cohort in cohorts
        ):
            msg = f"Attribute {attribute} not found in all cohorts."
            raise ValueError(msg)

        # get attribute values for tests
        samples = [
            extract_attribute_values(
                medrecord=cohort.medrecord,
                group=cohort.patient_group,
                attributes=[cohort.attributes[attribute]],
                nodes_or_edges="nodes",
            )[cohort.attributes[attribute]]
            for cohort in cohorts
        ]

        types = [
            cohort.attribute_summary[cohort.patient_group][attribute]["type"]
            for cohort in cohorts
        ]

        # check if attribute types differ between cohorts
        if not all(types[0] == attr_type for attr_type in types):
            msg = f"""Not all attribute types are the same, found types
                {", ".join(types)} for attribute {attribute}."""
            raise ValueError(msg)

        attr_type = AttributeType[types[0]] if types[0] != "Unstructured" else None

        # test difference if the attribute type could be determined
        if attr_type:
            return decide_hypothesis_test(
                samples,
                alpha=significance_level,
                attribute_type=attr_type,
            )

        return None

    @staticmethod
    def test_difference_concept_attribute(
        cohorts: List[CohortEvaluator],
        attribute: MedRecordAttribute,
        concept: Group,
        significance_level: float = 0.05,
    ) -> Optional[TestSummary]:
        """Use hypothesis test to test for difference in mean or distribution.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts to compare.
            attribute (MedRecordAttribute): Patient concept edge attribute to test.
            concept (Group): Concept that has the edge attribute.
            significance_level (float, optional): Significance level for the test.
                Defaults to 0.05.

        Raises:
            ValueError: If attribute not found in all cohorts.
            ValueError: If attribute has different type in different cohorts.

        Returns:
            Optional[TestSummary]: TestSummary if possible.
        """
        if not all(concept in cohort.attribute_summary for cohort in cohorts):
            msg = f"Concept {concept} not found in cohort attribute summary."
            raise ValueError(msg)

        if not all(
            attribute in cohort.attribute_summary[concept] for cohort in cohorts
        ):
            msg = f"Attribute {attribute} not found in all cohorts."
            raise ValueError(msg)

        # get attribute values for tests
        samples = [
            extract_attribute_values(
                medrecord=cohort.medrecord,
                group=cohort.concepts_edges[concept],
                attributes=[cohort.attributes[attribute]],
                nodes_or_edges="edges",
            )[cohort.attributes[attribute]]
            for cohort in cohorts
        ]

        types = [
            cohort.attribute_summary[concept][attribute]["type"] for cohort in cohorts
        ]

        # check if attribute types differ between cohorts
        if not all(types[0] == attr_type for attr_type in types):
            msg = f"""Not all attribute types are the same, found types
                {", ".join(types)} for attribute {attribute}."""
            raise ValueError(msg)

        attr_type = AttributeType[types[0]] if types[0] != "Unstructured" else None

        # test difference if the attribute type could be determined
        if attr_type:
            return decide_hypothesis_test(
                samples,
                alpha=significance_level,
                attribute_type=attr_type,
            )

        return None

    @staticmethod
    def plot_top_k_concepts(
        cohorts: List[CohortEvaluator],
        top_k: int = 10,
    ) -> Tuple[Figure, Axes]:
        """Plot the top k concepts for each cohort and return the figure and axis.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts to compare.
            top_k (int): Number of top concepts to show.

        Returns:
            Tuple[Figure, Axes]: Figure and axis of the plot.
        """
        # Get all concept ids of first cohort
        top_k_tuple = cohorts[0].get_top_k_concepts(top_k=top_k)
        top_k_ids = [concept[0] for concept in top_k_tuple]

        top_k_all = {
            cohort.name: {
                concept_id: freq / len(cohort.medrecord.edges) * 100
                for concept_id, freq in cohort.get_top_k_concepts(top_k=top_k * 10)
            }
            for cohort in cohorts
        }

        # Filter to only the top_k_ids from the first cohort
        cohort_frequencies = {
            cohort.name: {
                concept_id: top_k_all[cohort.name].get(concept_id, 0)
                for concept_id in top_k_ids
            }
            for cohort in cohorts
        }

        # Define bar width
        num_cohorts = len(cohorts)
        width = 0.8 / max(num_cohorts, 1)  # Prevent division by zero

        # Generate x-axis positions
        x = np.arange(len(top_k_ids))
        fig, ax = plt.subplots(figsize=(20, 6))

        for i, (name, frequencies) in enumerate(cohort_frequencies.items()):
            ax.bar(
                x + (i - num_cohorts / 2) * width,
                list(frequencies.values()),
                width=width,
                label=name,
                alpha=0.7,
            )

        # Set categorical labels
        ax.set_xticks(x)
        ax.set_xticklabels(top_k_ids, rotation=45, ha="right")  # pyright: ignore[reportArgumentType]

        # Labels and legend
        ax.set_xlabel("Concept")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Comparison of Top {top_k} Concepts Across Cohorts")
        ax.legend()

        # Return figure and axis for further customization
        return fig, ax

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
                "concept_attribute_info": cohort.medrecord._describe_group_edges(
                    groups=list(cohort.concepts_edges.values())
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
            control_stats = control_group.attribute_summary[
                control_group.patient_group
            ][attribute]
            case_stats = case_group.attribute_summary[case_group.patient_group][
                attribute
            ]

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
    def calculate_distance_concepts(
        real_data: CohortEvaluator, synthetic_data: CohortEvaluator
    ) -> Dict[Group, DistanceSummary]:
        """Calculate the distance between cohorts.

        Compute the distances btw. real and synthetic data based on
        the following stats: Jensen-Shannon-Divergence and normalized distance for
        patients (p) and visits (v).

        Args:
            real_data (CohortEvaluator): The real data for comparison.
            synthetic_data (CohortEvaluator): The synthesized data to compare.

        Returns:
            Dict[Group, DistanceSummary]: The distance summary for each concept.
        """
        ...

    @staticmethod
    def full_comparison(
        cohorts: List[CohortEvaluator],
        top_k: int,
        significance_level: float,
    ) -> Tuple[Dict[str, CohortSummary], ComparerSummary]:
        """A full comparison of all functions.

        Args:
            cohorts (List[CohortEvaluator]): List of cohorts to compare.
            top_k (int): Top k concepts to compare.
            significance_level (float): Significance level for the hypothesis tests.

        Returns:
            Tuple[Dict[str, CohortSummary], ComparerSummary]: Result of the full
                comparison for the cohorts.
        """
        ...


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
    attributes = set(cohorts[0].attribute_summary[cohorts[0].patient_group].keys())

    for i in range(1, len(cohorts) - 1):
        attributes = attributes.intersection(
            set(cohorts[i].attribute_summary[cohorts[i].patient_group].keys())
        )

    if len(attributes) == 0:
        msg = "No common attribute found between the cohorts."
        raise ValueError(msg)

    return list(attributes)

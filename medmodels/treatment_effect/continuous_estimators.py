"""Functions to estimate the treatment effect for continuous outcomes."""

import logging
from math import sqrt
from typing import Literal, Set

import numpy as np

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex
from medmodels.treatment_effect.temporal_analysis import find_reference_edge

logger = logging.getLogger(__name__)


def average_treatment_effect(
    medrecord: MedRecord,
    treatment_outcome_true_set: Set[NodeIndex],
    control_outcome_true_set: Set[NodeIndex],
    outcome_group: Group,
    outcome_variable: MedRecordAttribute,
    reference: Literal["first", "last"] = "last",
    time_attribute: MedRecordAttribute = "time",
) -> float:
    r"""Calculates the Average Treatment Effect (ATE).

    It is calculated as the difference between the outcome means of the treated and
    control sets. A positive ATE indicates that the treatment increased the outcome,
    while a negative ATE suggests a decrease.

    The ATE is computed as follows when the numbers of observations in treated and
    control sets are N and M, respectively:

    .. math::

        \\text{ATE} = \\frac{1}{N} \\sum_i y_1(i) - \\frac{1}{M} \\sum_j y_0(j),

    where :math:`y_1(i)` and :math:`y_0(j)` represent outcome values for individual
    treated and control observations. In the case of matched sets with equal sizes
    (N = M), the formula simplifies to:

    .. math::

        \\text{ATE} = \\frac{1}{N} \\sum_i (y_1(i) - y_0(i)).

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing medical
            data.
        treatment_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the treated group that also have the outcome.
        control_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the control group that have the outcome.
        outcome_group (Group): The group of nodes that contain the outcome variable.
        outcome_variable (MedRecordAttribute): The attribute in the edge that contains
            the outcome variable. It must be numeric and continuous.
        reference (Literal["first", "last"], optional): The reference point for the
            exposure time. Options include "first" and "last". If "first", the function
            returns the earliest exposure edge. If "last", the function returns the
            latest exposure edge. Defaults to "last".
        time_attribute (MedRecordAttribute, optional): The attribute in the edge that
            contains the time information. Defaults to "time".

    Returns:
        float: The average treatment effect.

    Raises:
        ValueError: If the outcome variable is not numeric.
    """
    treated_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in treatment_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in treated_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    control_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in control_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in control_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    return treated_outcomes.mean() - control_outcomes.mean()


def cohens_d(
    medrecord: MedRecord,
    treatment_outcome_true_set: Set[NodeIndex],
    control_outcome_true_set: Set[NodeIndex],
    outcome_group: Group,
    outcome_variable: MedRecordAttribute,
    reference: Literal["first", "last"] = "last",
    time_attribute: MedRecordAttribute = "time",
) -> float:
    """Calculates Cohen's D, the standardized mean difference between two sets.

    This measures the effect size of the difference between two outcome means.
    It's applicable for any two sets but is recommended for sets of the same size.
    Cohen's D indicates how many standard deviations the two groups differ by, with 1
    standard deviation equal to 1 z-score.

    A rule of thumb for interpreting Cohen's D:
    - Small effect = ±0.2
    - Medium effect = ±0.5
    - Large effect = ±0.8

    If the difference is negative, it indicates the mean in the treated group is lower
    than the control group.

    This metric provides a dimensionless measure of effect size, facilitating the
    comparison across different studies and contexts.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing medical
            data.
        treatment_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the treated group that also have the outcome.
        control_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the control group that have the outcome.
        outcome_group (Group): The group of nodes that contain the outcome variable.
        outcome_variable (MedRecordAttribute): The attribute in the edge that contains
            the outcome variable. It must be numeric and continuous.
        reference (Literal["first", "last"], optional): The reference point for the
            exposure time. Options include "first" and "last". If "first", the function
            returns the earliest exposure edge. If "last", the function returns the
            latest exposure edge. Defaults to "last".
        time_attribute (MedRecordAttribute, optional): The attribute in the edge that
            contains the time information. Defaults to "time".

    Returns:
        float: The Cohen's D coefficient, representing the effect size.

    Raises:
        ValueError: If the outcome variable is not numeric.

    Warning:
        If the sample size is small (less than 50), the function advises Hedges' g use.
    """
    treated_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in treatment_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in treated_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    control_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in control_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in control_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    minimum_length = min(len(treated_outcomes), len(control_outcomes))

    if minimum_length < 50:
        logger.warning(
            "Small sample size detected. Consider using Hedges' g for an unbiased effect size estimate."
        )

    return (treated_outcomes.mean() - control_outcomes.mean()) / sqrt(
        (treated_outcomes.std(ddof=1) ** 2 + control_outcomes.std(ddof=1) ** 2) / 2
    )


def hedges_g(
    medrecord: MedRecord,
    treatment_outcome_true_set: Set[NodeIndex],
    control_outcome_true_set: Set[NodeIndex],
    outcome_group: Group,
    outcome_variable: MedRecordAttribute,
    reference: Literal["first", "last"] = "last",
    time_attribute: MedRecordAttribute = "time",
) -> float:
    """Calculates Hedges' g, the unbiased effect size estimate.

    Hedges' g is a corrected version of Cohen's d that provides an unbiased estimate
    of the effect size, especially important when sample sizes are small (under 50).

    The correction factor is applied regardless of the sample size.

    Args:
        medrecord (MedRecord): An instance of the MedRecord class containing medical
            data.
        treatment_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the treated group that also have the outcome.
        control_outcome_true_set (Set[NodeIndex]): A set of node indices representing
            the control group that have the outcome.
        outcome_group (Group): The group of nodes that contain the outcome variable.
        outcome_variable (MedRecordAttribute): The attribute in the edge that contains
            the outcome variable. It must be numeric and continuous.
        reference (Literal["first", "last"], optional): The reference point for the
            exposure time. Options include "first" and "last". If "first", the function
            returns the earliest exposure edge. If "last", the function returns the
            latest exposure edge. Defaults to "last".
        time_attribute (MedRecordAttribute, optional): The attribute in the edge that
            contains the time information. Defaults to "time".

    Returns:
        float: The Hedges' g coefficient, representing the effect size.

    Raises:
        ValueError: If the outcome variable is not numeric.
    """
    treated_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in treatment_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in treated_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    control_outcomes = np.array(
        [
            medrecord.edge[
                find_reference_edge(
                    medrecord,
                    node_index,
                    outcome_group,
                    time_attribute=time_attribute,
                    reference=reference,
                )
            ][outcome_variable]
            for node_index in control_outcome_true_set
        ]
    )
    if not all(isinstance(i, (int, float)) for i in control_outcomes):
        msg = "Outcome variable must be numeric"
        raise ValueError(msg)

    number_treated = len(treated_outcomes)
    number_control = len(control_outcomes)
    degrees_of_freedom = number_treated + number_control - 2

    cohen_d = cohens_d(
        medrecord,
        treatment_outcome_true_set,
        control_outcome_true_set,
        outcome_group,
        outcome_variable,
        reference,
        time_attribute,
    )

    # Correction factor J
    correction_factor_j = 1 - (3 / (4 * degrees_of_freedom - 1))

    return correction_factor_j * cohen_d


CONTINUOUS_ESTIMATOR = {
    "average_te": average_treatment_effect,
    "cohens_d": cohens_d,
    "hedges_g": hedges_g,
}

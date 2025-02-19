"""Inferential stastics functions for analyzing data distributions."""

from typing import List, Optional, TypedDict

import numpy as np
import scipy
import scipy.stats

from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import MedRecordValue


class TestSummary(TypedDict):
    """Result of a hypothesis test."""

    test: str
    Hypothesis: str
    not_reject: bool
    p_value: float


def decide_hypothesis_test(
    samples: List[List[MedRecordValue]], alpha: float, attribute_type: AttributeType
) -> Optional[TestSummary]:
    """Entry point for testing hypothesis of similarity between samples.

    Args:
        samples (List[List[MedRecordValue]]): List of samples.
        alpha (float): Significance level for the tests. Has to be between 0 and 1.
        attribute_type (AttributeType): Type of attribute that should be compared.

    Returns:
        Optional[TestSummary]: Test Summary if possible for the amount of samples and
            the attribute type.

    Raises:
        ValueError: If significance level is not between 0 and 1.
    """
    if alpha < 0 or alpha > 1:
        msg = f"Sigificance level should be between 0 and 1, not {alpha}."
        raise ValueError(msg)

    if attribute_type == AttributeType.Continuous:
        # check for normal distribution
        if all(normal_distribution_test(sample) for sample in samples):
            if len(samples) == 2:
                return two_tailed_t_test(*samples, alpha=alpha)
            return analysis_of_variance(samples, alpha=alpha)
        if len(samples) == 2:
            return mann_whitney_u_test(*samples, alpha=alpha)

    if attribute_type == AttributeType.Categorical and len(samples) == 2:
        return chi_square_independece_test(*samples, alpha=alpha)

    return None


def normal_distribution_test(sample: List[MedRecordValue], alpha: float = 0.05) -> bool:
    """Check the null hypothesis that the values come from a normal distribution.

    Args:
        sample (List[MedRecordValue]): List of attribute values.
        alpha (float): Significance level for the test. Defaults to 0.05.

    Returns:
        bool: True if null hypothesis can not be rejected
    """
    result = scipy.stats.normaltest(np.array(sample), nan_policy="omit")

    not_reject = True

    return not_reject if result.pvalue > alpha else not not_reject


def two_tailed_t_test(
    sample1: List[MedRecordValue], sample2: List[MedRecordValue], alpha: float = 0.05
) -> TestSummary:
    """Tests for difference in means between two samples.

    Args:
        sample1 (List[MedRecordValue]): First sample group to test.
        sample2 (List[MedRecordValue]): Second sample to compare with the first.
        alpha (float): Significance level for the test. Defaults to 0.05.

    Returns:
        TestSummary: Summary of the test and its null hypothesis.
    """
    result = scipy.stats.ttest_ind(
        np.array(sample1), np.array(sample2), nan_policy="omit"
    )

    not_reject = True

    assert isinstance(result.pvalue, float)

    return {
        "test": "t-Test",
        "Hypothesis": """There is no significant difference between the means of the
         two populations.""",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def mann_whitney_u_test(
    sample1: List[MedRecordValue], sample2: List[MedRecordValue], alpha: float = 0.05
) -> TestSummary:
    """Test if two samples have the same underlying distribution.

    Args:
        sample1 (List[MedRecordValue]): First sample for testing.
        sample2 (List[MedRecordValue]): Second sample for testing.
        alpha (float, optional): Significance level of the test. Defaults to 0.05.

    Returns:
        TestSummary: Summary of the test and its null hypothesis.
    """
    result = scipy.stats.mannwhitneyu(
        np.array(sample1), np.array(sample2), nan_policy="omit"
    )

    not_reject = True

    assert isinstance(result.pvalue, float)

    return {
        "test": "Mann-Whitney U Test",
        "Hypothesis": "The distributions of both populations are equal.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def analysis_of_variance(
    samples: List[List[MedRecordValue]], alpha: float = 0.05
) -> TestSummary:
    """Test if the means of multiple samples are the same.

    Args:
        samples (List[List[MedRecordValue]]): List of two or more samples.
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        TestSummary: Summary of the test and its null hypothesis.

    Raises:
        ValueError: If less than two populations.
    """
    if len(samples) < 2:
        msg = "Need minimum two populations to test."
        raise ValueError(msg)
    sample_arrays = [np.array(sample) for sample in samples]
    result = scipy.stats.f_oneway(*sample_arrays, nan_policy="omit")

    not_reject = True

    assert isinstance(result.pvalue, float)

    return {
        "test": "ANOVA",
        "Hypothesis": "The means of all populations are equal.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def chi_square_independece_test(
    sample1: List[MedRecordValue], sample2: List[MedRecordValue], alpha: float = 0.05
) -> TestSummary:
    """Goodness-of-fit test for two categorical distributions.

    Args:
        sample1 (List[MedRecordValue]): First sample of categorical data.
        sample2 (List[MedRecordValue]): second sample of categorical data.
        alpha (float, optional): Significance level for the test. Defaults to 0.05.

    Returns:
        TestSummary: Summary of the test and its null hypothesis.

    Raises:
        ValueError: If categories are different in the samples.
    """
    if (set(sample1) - set(sample2)) or (set(sample2) - set(sample1)):
        msg = "Different categories are found for the samples."
        raise ValueError(msg)

    # get frequencies
    sample_set = sorted([str(value) for value in sample1])
    sample_compare = sorted([str(value) for value in sample2])

    freq1 = {x: sample_set.count(x) for x in sample_set}
    freq2 = {x: sample_compare.count(x) for x in sample_compare}

    proportions1 = [freq / sum(freq1.values()) for freq in freq1.values()]
    proportions2 = [freq / sum(freq2.values()) for freq in freq2.values()]

    result = scipy.stats.chisquare(np.array(proportions1), np.array(proportions2))

    not_reject = True

    assert isinstance(result.pvalue, float)

    return {
        "test": "Pearson's chi-squared Test",
        "Hypothesis": "The two samples are drawn from the same distribution.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def kolmogorov_smirnov_test(
    sample1: List[MedRecordValue], sample2: List[MedRecordValue], alpha: float
) -> TestSummary:
    """Compares two samples without prior assumption of their distribution.

    Args:
        sample1 (List[MedRecordValue]): First sample to compare.
        sample2 (List[MedRecordValue]): Second sample to compare.
        alpha (float): Significance level.

    Returns:
        TestSummary: Summary of the test and its null hypothesis.
    """
    # assume null hypothesis can not be rejected
    not_reject = True

    result = scipy.stats.ks_2samp(
        np.array(sample1), np.array(sample2), nan_policy="omit"
    )

    return {
        "test": "Kolmogorov-Smirnov Test",
        "Hypothesis": "The two samples are drawn from the same distribution.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


# def measure_effect_size(samples: List[MedRecordValue]) -> Tuple[str, float]: ...


def calculate_relative_difference(control_mean: float, case_mean: float) -> float:
    """Calculates the absolute relative difference.

    Calculates the absolute relative difference for a single feature, expressed as a
    percentage of the control's mean. Handles division by zero by returning the
    absolute difference when the control mean is zero.

    Args:
        control_mean (float): Mean of the feature for the control group.
        case_mean (float): Mean of the feature for the treated group.

    Returns:
        float: Absolute relative difference.
    """
    if control_mean == 0:
        return abs(case_mean - control_mean) * 100
    return abs((case_mean - control_mean) / control_mean) * 100

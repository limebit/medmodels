# ruff: noqa: D100, D103, T201
from typing import List, Optional, Tuple

import numpy as np
import scipy
import scipy.stats

from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import MedRecordValue
from medmodels.statistic_evaluations.comparer.data_comparer import TestSummary


def decide_hypothesis_test(
    samples: List[List[MedRecordValue]], alpha: float, attribute_type: AttributeType
) -> Optional[TestSummary]:
    """Entry point for testing hypothesis of similarity between samples.

    Args:
        samples (List[List[MedRecordValue]]): List of samples.
        alpha (float): Significance level for the tests.
        attribute_type (AttributeType): Type of attribute that should be compared.

    Returns:
        Optional[TestSummary]: Test Summary if possible for the amount of samples and
            the attribute type.
    """
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
    result = scipy.stats.normaltest(sample, nan_policy="omit")

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

    return {
        "test": "t-Test",
        "Hypothesis": """There is no significant difference between the means of the
         two populations.""",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.p_value > alpha else not not_reject,
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
    result = scipy.stats.mannwhitneyu(sample1, sample2, nan_policy="omit")

    not_reject = True

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
    result = scipy.stats.f_oneway(*samples, nan_policy="omit")

    not_reject = True

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
    """
    if (set(sample1) - set(sample2)) or (set(sample2) - set(sample1)):
        msg = "Different categories are found for the samples."
        raise ValueError(msg)

    # get frequencies
    sample_set = sorted([str(value) for value in sample1])
    sample_compare = sorted([str(value) for value in sample2])

    freq1 = {x: sample_set.count(x) for x in sample_set}
    freq2 = {x: sample_compare.count(x) for x in sample_compare}

    result = scipy.stats.chisquare(freq1.values(), freq2.values())

    not_reject = True

    return {
        "test": "Pearson's chi-squared Test",
        "Hypothesis": "The two samples are drawn from the same distribution.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def kolmogorov_smirnov_test(
    sample1: List[MedRecordValue], sample2: List[MedRecordValue], alpha: float
) -> TestSummary:
    # assume null hypothesis can not be rejected
    not_reject = True

    result = scipy.stats.ks_2samp(sample1, sample2, nan_policy="omit")

    return {
        "test": "Kolmogorov-Smirnov Test",
        "Hypothesis": "The two samples are drawn from the same distribution.",
        "p_value": result.pvalue,
        "not_reject": not_reject if result.pvalue > alpha else not not_reject,
    }


def measure_effect_size(samples: List[MedRecordValue]) -> Tuple[str, float]: ...

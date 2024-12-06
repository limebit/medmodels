from typing import List, Tuple

from numpy.typing import ArrayLike
import scipy

from medmodels.statistic_evaluations.comparer.data_comparer import TestSummary


def decide_hypothesis_test(samples: List[ArrayLike], alpha: float) -> TestSummary:
    """Entry

    Args:
        samples (List[ArrayLike]): _description_
        alpha (float): _description_

    Returns:
        TestSummary: _description_
    """

    # get attribute type

    attribute_type = ""  # get type from function

    if attribute_type == "Continuous":
        # check for normal distribution
        if all(normal_distribution_test(sample) for sample in samples):
            if len(samples) == 2:
                return two_tailed_t_test(samples, alpha)
            else:
                return analysis_of_variance(samples, alpha)
        else:
            # choose test that does not need normal distribution
            if len(samples) == 2:
                return mann_whitney_u_test(samples, alpha)

    if attribute_type == "Categorical":
        return chi_square_independece_test(samples, alpha)


def normal_distribution_test(sample: ArrayLike) -> bool: ...


def two_tailed_t_test(samples: List[ArrayLike], alpha: float) -> TestSummary:
    if len(samples) != 2:
        raise ValueError("T-Test is only possible for comparing two populations.")
    _, p_value = scipy.stats.ttest_ind(*samples)

    test_summary = {
        "test": "t-Test",
        "Hypothesis": """There is no significant difference between the means of the
         two populations.""",
        "p_value": p_value,
    }

    if p_value > alpha:
        test_summary["not_reject"] = False
    else:
        test_summary["not_reject"] = True

    return test_summary


def mann_whitney_u_test(samples: List[ArrayLike], alpha: float) -> TestSummary:
    if len(samples) != 2:
        raise ValueError(
            "Mann Whitney U Test is only possible for comparing two populations."
        )
    _, p_value = scipy.stats.mannwhitneyu(*samples)

    test_summary = {
        "test": "Mann-Whitney U Test",
        "Hypothesis": "The distributions of both populations are equal.",
        "p_value": p_value,
    }

    if p_value > alpha:
        test_summary["not_reject"] = False
    else:
        test_summary["not_reject"] = True

    return test_summary


def analysis_of_variance(samples: List[ArrayLike], alpha: float) -> TestSummary:
    _, p_value = scipy.stats.f_oneway(*samples)

    test_summary = {
        "test": "ANOVA",
        "Hypothesis": "The means of all populations are equal.",
        "p_value": p_value,
    }

    if p_value > alpha:
        test_summary["not_reject"] = False
    else:
        test_summary["not_reject"] = True

    return test_summary


def chi_square_independece_test(
    samples: List[ArrayLike], alpha: float
) -> TestSummary: ...
def kolmogorov_smirnov_test(samples: List[ArrayLike], alpha: float):
    if len(samples) != 2:
        raise ValueError(
            "Kolmogorov-Smirnov Test is only possible for comparing two populations."
        )
    _, p_value = scipy.stats.ks_2amp(*samples)

    test_summary = {
        "test": "Kolmogorov-Smirnov Test",
        "Hypothesis": "The two samples are drawn from the same distribution.",
        "p_value": p_value,
    }

    if p_value > alpha:
        test_summary["not_reject"] = False
    else:
        test_summary["not_reject"] = True

    return test_summary


def measure_effect_size(samples: List[ArrayLike]) -> Tuple[str, float]: ...

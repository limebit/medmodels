import logging
import pandas as pd
from statistics import mean, stdev
from math import sqrt


logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(format="%(message)s")


def average_treatment_effect(
    treated_set: pd.DataFrame, control_set: pd.DataFrame, outcome_variable: str
):

    """
    Calculates an average treatment effect as a difference between the outcome means of
    the treated and control sets. Positive TE means that he treatment increased the
    outcome, negative - decreased.

    Because for one individual not both treated and control outcomes can be measured,
    the ATE for a treated set with N and a control set with M elements is computed as
    follows:
    .. math::
         \text{ATE} = \frac {1}{N} \\sum_i y_1(i) - \frac {1}{M} \\sum_j y_0(j),

    where y_1(i), y_0(j) are outcome values for one treated and control observation each

    For the matched set the formula stays:

    .. math::
         \text{ATE} = \frac {1}{N} \\sum_i y_1(i) - \frac {1}{M} \\sum_j y_0(j)
                    = \frac {1}{N} (\\sum_i y_1(i) - y_0(i)) if N = M

    :param treated_set: treated set of patients
    :type treated_set: pd.DataFrame
    :param control_set: control set of patients
    :type control_set: pd.DataFrame
    :param outcome_variable: outcome variable
    :type outcome_variable: str
    :return: average treatment effect
    :rtype: float
    """
    return treated_set[outcome_variable].mean() - control_set[outcome_variable].mean()


def cohen_d(
    treated_set: pd.DataFrame,
    control_set: pd.DataFrame,
    outcome_variable: str,
    add_correction: bool = False,
):
    """
    Calculates the Cohen’s D (standardized mean difference). It measures the effect size
    of the difference between two outcome means. Can be computed for any two sets but
    recommended for sets with same amount of elements.

    A d of i indicates the two groups differ by i standard deviations. Standard
    deviations are equivalent to z-scores (1 standard deviation = 1 z-score).
    Rule of thumb:
            Small effect = 0.2
            Medium Effect = 0.5
            Large Effect = 0.8

    :param treated_set: treated set
    :type treated_set: pd.DataFrame
    :param control_set: control set
    :type control_set: pd.DataFrame
    :param outcome_variable: outcome variable
    :type outcome_variable: str
    :param add_correction: correction factor for small groups. Defaults to False.
    :type add_correction: bool, optional
    :return: cohen's D coefficient
    :rtype: float
    """
    min_len = min(treated_set.shape[0], control_set.shape[0])

    # correction factor
    cf = 1

    if min_len < 50:
        logging.warning(
            "For sets with size < 50 better use the alternative method "
            "Hedges' g, which provides a slightly smaller bias towards "
            "small samples."
        )
        # correction factor
        if add_correction:
            cf = (min_len - 3) * sqrt((min_len - 2) / min_len) / (min_len - 2.25)

    c0 = treated_set[outcome_variable].to_list()
    c1 = control_set[outcome_variable].to_list()

    return (mean(c0) - mean(c1)) * cf / sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2)


ESTIMATORS_CONT = {
    "average_te": average_treatment_effect,
    "cohen_d": cohen_d,
}

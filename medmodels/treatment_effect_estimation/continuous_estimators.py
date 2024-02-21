import logging
from math import sqrt
from statistics import mean, stdev

import pandas as pd

logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(format="%(message)s")


def average_treatment_effect(
    treated_set: pd.DataFrame, control_set: pd.DataFrame, outcome_variable: str
) -> float:
    """
    Calculates the Average Treatment Effect (ATE) as the difference between the outcome
    means of the treated and control sets. A positive ATE indicates that the treatment
    increased the outcome, while a negative ATE suggests a decrease.

    The ATE is computed as follows when the numbers of observations in treated and
    control sets are N and M, respectively:

    $$
    \text{ATE} = \frac{1}{N} \sum_i y_1(i) - \frac{1}{M} \sum_j y_0(j),
    $$

    where $y_1(i)$ and $y_0(j)$ represent outcome values for individual treated and
    control observations. In the case of matched sets with equal sizes (N = M), the
    formula simplifies to:

    $$
    \text{ATE} = \frac{1}{N} \sum_i (y_1(i) - y_0(i)).
    $$

    Args:
        treated_set (pd.DataFrame): DataFrame of the treated group.
        control_set (pd.DataFrame): DataFrame of the control group.
        outcome_variable (str): Name of the outcome variable.

    Returns:
        float: The average treatment effect.

    This function provides a simple yet powerful method for estimating the impact of a
    treatment by comparing average outcomes between treated and control groups.
    """

    return treated_set[outcome_variable].mean() - control_set[outcome_variable].mean()


def cohen_d(
    treated_set: pd.DataFrame,
    control_set: pd.DataFrame,
    outcome_variable: str,
    add_correction: bool = False,
) -> float:
    """
    Calculates Cohen's D, the standardized mean difference between two sets, measuring
    the effect size of the difference between two outcome means. It's applicable for
    any two sets but is recommended for sets of the same size. Cohen's D indicates how
    many standard deviations the two groups differ by, with 1 standard deviation equal
    to 1 z-score.

    A rule of thumb for interpreting Cohen's D:
    - Small effect = 0.2
    - Medium effect = 0.5
    - Large effect = 0.8

    Args:
        treated_set (pd.DataFrame): DataFrame containing the treated group data.
        control_set (pd.DataFrame): DataFrame containing the control group data.
        outcome_variable (str): The name of the outcome variable to analyze.
        add_correction (bool, optional): Whether to apply a correction factor for small
            sample sizes. Defaults to False.

    Returns:
        float: The Cohen's D coefficient, representing the effect size.

    This metric provides a dimensionless measure of effect size, facilitating the
    comparison across different studies and contexts.
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

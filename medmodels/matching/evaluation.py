from __future__ import annotations

from typing import List, Optional, Tuple, Union

import pandas as pd


def calculate_relative_diff(row: pd.Series[float]) -> float:
    """
    Calculates the absolute relative difference for a single feature, expressed as a
    percentage of the control's mean. Handles division by zero by returning the
    absolute difference when the control mean is zero.
    Args:
        row (pd.Series[float]): A Series object representing a row from the DataFrame of
            means, containing 'control_mean' and 'treated_mean' for a feature.
    Returns:
        float: The absolute relative difference in means, as a percentage.
    """

    control_mean = row["control_mean"]
    treated_mean = row["treated_mean"]

    if control_mean == 0:
        return abs(treated_mean - control_mean) * 100
    else:
        return abs((treated_mean - control_mean) / control_mean) * 100


def relative_diff_in_means(
    control_set: pd.DataFrame, treated_set: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates the absolute relative mean difference for each feature between control
    and treated sets, expressed as a percentage of the control set's mean. This measure
    provides an understanding of how much each feature's average value changes from the
    control to the treated group relative to the control.

    Args:
        control_set (pd.DataFrame): DataFrame representing the control group.
        treated_set (pd.DataFrame): DataFrame representing the treated group.

    Returns:
        pd.DataFrame: A DataFrame containing the mean values of the control and treated
            sets for all features and the absolute relative difference in means,
            expressed as a percentage.

    The function internally computes the relative difference for each feature, handling
    cases where the control mean is zero by simply calculating the absolute difference
    times 100. It provides insights into the percentage change in feature means due to
    treatment.
    """

    control_mean = pd.DataFrame(control_set.mean()).transpose()
    treated_mean = pd.DataFrame(treated_set.mean()).transpose()
    df_mean = pd.concat([control_mean, treated_mean], ignore_index=True)

    df_mean = df_mean.rename(index={0: "control_mean", 1: "treated_mean"})
    df_mean = df_mean.transpose()
    df_mean["Diff (in %)"] = df_mean.apply(calculate_relative_diff, axis=1)

    return df_mean.transpose()


def average_value_over_features(df: pd.DataFrame) -> float:
    """
    Calculates the average of the values in the last row of a DataFrame. This function
    is particularly useful for aggregating measures like differences or percentages
    across multiple features, providing a single summary statistic.

    Args:
        df (pd.DataFrame): The DataFrame on which the calculation is to be performed.

    Returns:
        float: The average value of the last row across all columns.

    Example:
        Given a DataFrame with the last row containing differences in percentages
        between treated and control means across features 'a' and 'b', e.g., 75.0% for
        'a' and 250.0% for 'b', this function will return the average difference, which
        is (75.0 + 250.0) / 2 = 162.5.
    """

    return df.tail(1).mean(axis=1).values.tolist()[0]


def average_abs_relative_diff(
    control_set: pd.DataFrame,
    treated_set: pd.DataFrame,
    covariates: Optional[Union[List[str], pd.Index[str]]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Calculates the average absolute relative difference in means over specified
    covariates between control and treated sets. If covariates are not specified, the
    calculation includes all features.

    This function is designed to assess the impact of a treatment across multiple
    features by computing the mean of absolute relative differences. It returns both a
    summary metric and a detailed DataFrame for further analysis.

    Args:
        control_set (pd.DataFrame): DataFrame for the control group.
        treated_set (pd.DataFrame): DataFrame for the treated group.
        covariates (Optional[Union[List[str], pd.Index[str]]] optional): List
            of covariate names to include. If None, considers all features.

    Returns:
        Tuple[float, pd.DataFrame]: A tuple containing the average absolute relative
            difference as a float and a DataFrame with detailed mean values and absolute
            relative differences for all features.

    The detailed DataFrame includes means for both control and treated sets and the
    absolute relative difference for each feature.
    """

    if not covariates:
        covariates = treated_set.columns

    df_mean = relative_diff_in_means(control_set, treated_set)

    return average_value_over_features(df_mean[covariates]), df_mean

import pandas as pd
from medmodels.matching import metrics
import numpy as np
from pandas import DataFrame
from typing import Optional, List


def nearest_neighbor(
    treated_set: DataFrame,
    control_set: DataFrame,
    metric: str,
    covariates: Optional[List[str]] = None,
) -> DataFrame:
    """
    Performs nearest neighbor matching between two dataframes using a specified metric.
    This method employs a greedy algorithm to pair elements from the treated set with
    their closest matches in the control set based on the given metric. The algorithm
    does not optimize for the best overall matching but ensures a straightforward,
    commonly used approach. The method is flexible to different metrics and requires
    preliminary size comparison of treated and control sets to determine the direction
    of matching. It supports optional specification of covariates for focused matching.

    Args:
        treated_set (DataFrame): DataFrame for which matches are sought.
        control_set (DataFrame): DataFrame from which matches are selected.
        metric (str): Metric to measure closeness between units, e.g., "absolute",
            "mahalanobis".
        covariates (Optional[List[str]], optional): Covariates considered for matching.
            Defaults to all variables.

    Returns:
        DataFrame: Matched subset from the control set.
    """

    metric_function = metrics.METRICS[metric]
    columns = treated_set.columns

    if not covariates:
        covariates = columns

    treated_array = treated_set[covariates].to_numpy().astype(float)
    control_array = control_set[covariates].to_numpy().astype(float)
    control_array_full = control_set.to_numpy()  # To keep all the infos
    matched_group = pd.DataFrame(columns=columns)

    for element_ss in treated_array:

        dist = []

        if metric == "mahalanobis":
            # Calculate the covariance matrix
            cov = np.cov(np.concatenate((treated_array, control_array)).T)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.array([1 / cov])  # For the 1D case

            for element_bs in control_array:
                dist.append(metric_function(element_ss, element_bs, inv_cov=inv_cov))
        else:

            for element_bs in control_array:
                dist.append(metric_function(element_ss, element_bs))

        nn_index = np.argmin(dist)

        new_row = pd.DataFrame(control_array_full[nn_index], index=columns)
        matched_group = pd.concat([matched_group, new_row.transpose().astype(float)])
        # For the k:1 matching don't consider the chosen row any more.
        control_array_full = np.delete(control_array_full, nn_index, 0)
        control_array = np.delete(control_array, nn_index, 0)

    return matched_group.reset_index(drop=True)


ALGORITHMS = {"nearest neighbor": nearest_neighbor}

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from medmodels.matching import metrics
from medmodels.medrecord.types import MedRecordAttributeInputList


def nearest_neighbor(
    treated_set: pl.DataFrame,
    control_set: pl.DataFrame,
    metric: metrics.Metric,
    number_of_neighbors: int = 1,
    covariates: Optional[MedRecordAttributeInputList] = None,
) -> pl.DataFrame:
    """
    Performs nearest neighbor matching between two dataframes using a specified metric.
    This method employs a greedy algorithm to pair elements from the treated set with
    their closest matches in the control set based on the given metric. The algorithm
    does not optimize for the best overall matching but ensures a straightforward,
    commonly used approach. The method is flexible to different metrics and requires
    preliminary size comparison of treated and control sets to determine the direction
    of matching. It supports optional specification of covariates for focused matching.

    Args:
        treated_set (pl.DataFrame): DataFrame for which matches are sought.
        control_set (pl.DataFrame): DataFrame from which matches are selected.
        metric (metrics.Metric): Metric to measure closeness between units, e.g.,
            "absolute", "mahalanobis". The metric must be available in the metrics
            module.
        number_of_neighbors (int, optional): Number of nearest neighbors to find for
            each treated unit. Defaults to 1.
        covariates (Optional[MedRecordAttributeInputList], optional): Covariates
            considered for matching. Defaults to all variables.

    Returns:
        pl.DataFrame: Matched subset from the control set.
    """
    if not covariates:
        covariates = treated_set.columns

    treated_array = treated_set.select(covariates).to_numpy().astype(float)
    control_array = control_set.select(covariates).to_numpy().astype(float)
    control_array_full = control_set.to_numpy()  # To keep all the information
    matched_control = []

    cov = np.array([])
    if metric == "mahalanobis":
        cov = np.cov(np.concatenate((treated_array, control_array)).T)

    for treated_subject in treated_array:
        if metric == "mahalanobis":
            if cov.ndim == 0:
                inv_cov = 1 / cov
            else:
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "The covariance matrix is singular. Please, check the data."
                    )

            dist = [
                metrics.mahalanobis_metric(
                    treated_subject, control_subject, inv_cov=inv_cov
                )
                for control_subject in control_array
            ]

        else:
            metric_function = metrics.METRICS[metric]

            dist = [
                metric_function(treated_subject, control_subject)
                for control_subject in control_array
            ]

        neighbor_indices = np.argpartition(dist, number_of_neighbors)[
            :number_of_neighbors
        ]

        for neighbor_index in neighbor_indices:
            new_row = pl.DataFrame(
                [control_array_full[neighbor_index]], schema=treated_set.columns
            )
            matched_control.append(new_row)

            # For the k:1 matching don't consider the chosen row any more.
            control_array_full = np.delete(control_array_full, neighbor_index, 0)
            control_array = np.delete(control_array, neighbor_index, 0)

    return pl.concat(matched_control, how="vertical")


ALGORITHMS = {"nearest neighbor": nearest_neighbor}

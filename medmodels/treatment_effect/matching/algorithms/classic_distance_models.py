"""Models for matching treated and control units based on distance metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from medmodels.medrecord.types import MedRecordAttributeInputList


def nearest_neighbor(
    treated_set: pl.DataFrame,
    control_set: pl.DataFrame,
    number_of_neighbors: int = 1,
    covariates: Optional[MedRecordAttributeInputList] = None,
) -> pl.DataFrame:
    """Performs nearest neighbor matching between two polars dataframes.

    This method matches elements from the treated set with their closest matches in the
    control set based on the specified covariates. The function leverages the Hungarian
    algorithm (`linear_sum_assignment`) to find the optimal matching and ensures that
    each treated unit is matched with the specified number of nearest control units
    without reusing any control unit.

    Args:
        treated_set (pl.DataFrame): DataFrame for which matches are sought.
        control_set (pl.DataFrame): DataFrame from which matches are selected.
        number_of_neighbors (int, optional): Number of nearest neighbors to find for
            each treated unit. Defaults to 1.
        covariates (Optional[MedRecordAttributeInputList], optional): Covariates
            considered for matching. Defaults to all variables.

    Returns:
        pl.DataFrame: DataFrame containing the matched control units for each treated
            unit.

    Raises:
        ValueError: If the treated set is too large for the given number of neighbors.
    """
    if treated_set.shape[0] * number_of_neighbors > control_set.shape[0]:
        msg = "The treated set is too large for the given number of neighbors"
        raise ValueError(msg)

    if not covariates:
        covariates = treated_set.columns

    control_array, treated_array = normalize_data(control_set, treated_set, covariates)
    cost_matrix = np.linalg.norm(treated_array[:, np.newaxis] - control_array, axis=2)
    final_matches = np.full(
        (treated_array.shape[0], number_of_neighbors), -1, dtype=int
    )

    for neighbor_index in range(number_of_neighbors):
        # Solve the assignment problem using the Hungarian algorithm
        row_indices, column_indices = linear_sum_assignment(cost_matrix)

        # Store the matched indices
        final_matches[np.array(row_indices), neighbor_index] = column_indices

        # Remove matched controls from the cost matrix to avoid re-selection
        cost_matrix[:, column_indices] = np.inf

    # Flatten the final matches to get unique control indices
    matched_indices = final_matches.flatten()
    matched_indices = matched_indices[matched_indices != -1]

    return pl.DataFrame(
        control_set[matched_indices],
        orient="row",
        schema=control_set.columns,
    )


def normalize_data(
    control_set: pl.DataFrame,
    treated_set: pl.DataFrame,
    covariates: MedRecordAttributeInputList,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalizes the data of the control and treated sets on the specified covariates.

    The values of the specified covariates on the control and treated sets are
    normalized. This is performed by taking the maximum and minimum values of the
    combined treated and control sets and subtracting the minimum values from the
    data and dividing by the difference between the maximum and minimum values. The
    normalized data is then returned in the form of numpy arrays.

    Args:
        control_set (pl.DataFrame): Control set.
        treated_set (pl.DataFrame): Treated set.
        covariates (MedRecordAttributeInputList): Covariates to be normalized.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: Normalized control and treated
            arrays.
    """
    control_array = control_set.select(covariates).to_numpy().astype(float)
    treated_array = treated_set.select(covariates).to_numpy().astype(float)

    # Take the maximums and minimums per columns of both arrays combined
    maximum_values = np.maximum(
        np.max(treated_array, axis=0), np.max(control_array, axis=0)
    )
    minimum_values = np.minimum(
        np.min(treated_array, axis=0), np.min(control_array, axis=0)
    )

    # Normalize the data
    control_array = (control_array - minimum_values) / (maximum_values - minimum_values)
    treated_array = (treated_array - minimum_values) / (maximum_values - minimum_values)

    return control_array, treated_array

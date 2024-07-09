from typing import Optional, Tuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from medmodels.medrecord.types import MedRecordAttributeInputList


def nearest_neighbor(
    treated_set: pl.DataFrame,
    control_set: pl.DataFrame,
    number_of_neighbors: int = 1,
    covariates: Optional[MedRecordAttributeInputList] = None,
) -> pl.DataFrame:
    """
    Performs nearest neighbor matching between two dataframes using the Hungarian
    algorithm.

    This method matches elements from the treated set with their closest matches in the
    control set based on the specified covariates. The function leverages the Hungarian
    algorithm (linear_sum_assignment) to find the optimal matching and ensures that each
    treated unit is matched with the specified number of nearest control units without
    reusing any control unit.

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
    """
    if treated_set.shape[0] > control_set.shape[0] * number_of_neighbors:
        raise ValueError(
            "The treated set is too large for the given number of neighbors."
        )
    if not covariates:
        covariates = treated_set.columns

    control_array, treated_array = normalize_data(control_set, treated_set, covariates)

    # Create the cost matrix
    cost_matrix = np.linalg.norm(treated_array[:, np.newaxis] - control_array, axis=2)

    # Initialize a list to store the final matches and the used control indices
    used_control_indices = set()
    final_matches = np.full(
        (treated_array.shape[0], number_of_neighbors), -1, dtype=int
    )

    for i in range(number_of_neighbors):
        # Solve the assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out already used control indices
        row_ind_filtered = []
        col_ind_filtered = []

        for r, c in zip(row_ind, col_ind):
            if c not in used_control_indices:
                row_ind_filtered.append(r)
                col_ind_filtered.append(c)
                used_control_indices.add(c)

        # Store the matched indices
        final_matches[np.array(row_ind_filtered), i] = col_ind_filtered

        # Remove matched controls from the cost matrix to avoid re-selection
        cost_matrix[:, col_ind_filtered] = np.inf

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
    """
    Normalizes the data by taking the maximum and minimum values of the combined
    treated and control sets.

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

    # Normalize data: take the maximums and minimums per columns of both arrays combined
    max_vals = np.maximum(np.max(treated_array, axis=0), np.max(control_array, axis=0))
    min_vals = np.minimum(np.min(treated_array, axis=0), np.min(control_array, axis=0))

    # Normalize the data
    control_array = (control_array - min_vals) / (max_vals - min_vals)
    treated_array = (treated_array - min_vals) / (max_vals - min_vals)

    return control_array, treated_array

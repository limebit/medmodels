from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import polars as pl
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

from medmodels.medrecord.types import MedRecordAttributeInputList

Metric: TypeAlias = Literal[
    "minkowski",
    "euclidean",
    "cosine",
    "cityblock",
    "l1",
    "l2",
    "manhattan",
    "haversine",
    "nan_euclidean",
]

NNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]


def nearest_neighbor(
    treated_set: pl.DataFrame,
    control_set: pl.DataFrame,
    metric: Metric = "minkowski",
    algorithm: NNAlgorithm = "auto",
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
        metric (Metric): Metric to measure closeness between units, e.g.,
            "minkowski", "euclidean", "cosine", "cityblock", "l1", "l2", "manhattan",
            "haversine", "nan_euclidean". Defaults to "minkowski".
        algorithm (NNAlgorithm, optional): Algorithm used to compute nearest neighbors.
            Defaults to "auto". Options: "auto", "ball_tree", "kd_tree", "brute".
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
    control_array_full = control_set.to_numpy()

    nn = NearestNeighbors(
        n_neighbors=number_of_neighbors, metric=metric, algorithm=algorithm
    )
    nn.fit(control_array)
    _, indices = nn.kneighbors(treated_array)

    return pl.DataFrame(
        [
            control_array_full[neighbor_index]
            for neighbor_indices in indices
            for neighbor_index in neighbor_indices
        ],
        orient="row",
        schema=treated_set.columns,
    )


ALGORITHMS = {"nearest neighbor": nearest_neighbor}

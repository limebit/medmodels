"""Metrics for comparing vectors in the context of matching classes."""

import math

import numpy as np
from numpy.typing import NDArray


def absolute_metric(
    vector1: NDArray[np.float64], vector2: NDArray[np.float64]
) -> float:
    """
    Calculates the Manhattan distance (L1 norm) between two vectors, providing a measure
    of the absolute difference between them. This distance is the sum of the absolute
    differences between each corresponding pair of elements in the two vectors.

    Args:
        vector1 (NDArray[np.float64]): The first vector to be compared.
        vector2 (NDArray[np.float64]): The second vector to be compared.

    Returns:
        float: The Manhattan distance between the two vectors.

    The calculation is based on the formula:
        $$
        D(x, y) = ||x - y||_1 = \\sum_{i=1}^n |x_i - y_i| for x, y \\in \\mathbb{R}^n
        $$
    """

    diff = vector1 - vector2

    return sum(np.abs(diff))


def exact_metric(vector1: NDArray[np.float64], vector2: NDArray[np.float64]) -> float:
    """
    Computes the exact metric for matching, which is particularly applicable for
    discrete or categorical covariates rather than continuous ones. This metric returns
    0 if the two vectors are exactly identical, and infinity otherwise, making it
    suitable for scenarios where exact matches are necessary.

    The exact metric is defined as:

    $$
    D(x, y) =
    \\begin{cases}
    0      & \\text{if } x = y, \\\\
    \\infty & \\text{otherwise}.
    \\end{cases}
    $$

    Args:
        vector1 (NDArray[np.float64]): The first vector to be compared.
        vector2 (NDArray[np.float64]): The second vector to be compared.

    Returns:
        float: 0 if the vectors are equal, infinity if they are not.

    Note:
        This function is designed for exactly two input vectors.
    """
    if np.array_equal(vector1, vector2):
        return 0

    return np.inf


def mahalanobis_metric(*vectors: NDArray[np.float64], inv_cov: NDArray[np.float64]):
    """
    Returns mahalanobis metric for matching. Works better with continuous covariates.

    .. math::
        D(x, y) = \\sqrt{(x-y)^T S^{-1} (x-y)} where S is the covariate matrix of
        the whole distribution

    Args:
        vectors (NDArray[np.float64]): Two numpy arrays to be compared. They must
            belong to the same distribution.
        inv_cov (NDArray[np.float64]): The inverse of the covariance matrix of the
            whole distribution (data set).

    Returns:
        float: The Mahalanobis distance between the two vectors.

    The covariance matrix and its inverse are calculated at most ones per item to be
    paired, hence, they won't be included inside of the method in order to avoid the
    repeated computation.

    By matching without replacement the found paired item will be removed from the set,
    hence,the covariance matrix of the whole distribution and its inverse need to be
    recalculated for every entry. This can be time consuming for big data sets (esp.
    with a big amount of features).
    """

    diff = vectors[0] - vectors[1]
    left = np.dot(diff, inv_cov)
    right = np.dot(left, diff)

    if isinstance(right, np.ndarray):
        right = right.item()

    return float(math.sqrt(right))


METRICS = {
    "absolute": absolute_metric,
    "exact": exact_metric,
}

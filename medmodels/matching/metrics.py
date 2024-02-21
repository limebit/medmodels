from typing import Tuple
import numpy as np
import math


def absolute_metric(*vectors: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculates the Euclidean distance (L1 norm) between two vectors, providing a measure
    of the absolute difference between them. This distance is the sum of the absolute
    differences between each corresponding pair of elements in the two vectors.

    Args:
        vector1 (np.ndarray): The first vector for comparison.
        vector2 (np.ndarray): The second vector for comparison.

    Returns:
        float: The Euclidean distance between the two vectors.

    The calculation is based on the formula:
        $D(x, y) = ||x - y||_1 = \\sum_{i=1}^n |x_i - y_i| for x, y \\in \\mathbb{R}^n$
    """

    diff = vectors[0] - vectors[1]

    return sum(np.abs(diff))


def exact_metric(*vectors: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Computes the exact metric for matching, which is particularly applicable for
    discrete or categorical covariates rather than continuous ones. This metric returns
    0 if the two vectors are exactly identical, and infinity otherwise, making it
    suitable for scenarios where exact matches are necessary.

    The exact metric is defined as:

    $$
    D(x, y) =
    \begin{cases}
    0      & \text{if } x = y, \\
    \infty & \text{otherwise}.
    \end{cases}
    $$

    Args:
        vectors (Tuple[np.ndarray, np.ndarray]): Two numpy arrays to be compared.

    Returns:
        float: 0 if the vectors are equal, infinity if they are not.

    Note:
        This function is designed for exactly two input vectors.
    """
    v1, v2 = vectors[0], vectors[1]
    if np.array_equal(v1, v2):
        return 0
    else:
        return np.inf


def mahalanobis_metric(
    *vectors: Tuple[np.ndarray, np.ndarray], inv_cov: np.ndarray
) -> float:
    """
    Computes the Mahalanobis distance between two vectors, optimal for matching with
    continuous covariates. This distance considers the covariance among variables,
    making it effective for scale-invariant matching and variables of different scales.

    The Mahalanobis distance is defined as:

    $$
    D(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
    $$

    where $S$ is the covariance matrix of the entire distribution.

    Args:
        vectors (Tuple[np.ndarray, np.ndarray]): Two numpy arrays representing vectors
            from the same distribution to be compared.
        inv_cov (np.ndarray): The inverse of the covariance matrix of the entire
            distribution.

    Returns:
        float: The Mahalanobis distance between the vectors.

    Note:
        The covariance matrix and its inverse are calculated once per pairing to
        minimize redundant computations. For matching without replacement, the removal
        of paired items necessitates recalculating these matrices for each new pair,
        which can be computationally intensive for large datasets with many features.
    """

    diff = vectors[0] - vectors[1]
    left = np.dot(diff, inv_cov)
    right = np.dot(left, diff)
    return math.sqrt(right)


METRICS = {
    "absolute": absolute_metric,
    "exact": exact_metric,
    "mahalanobis": mahalanobis_metric,
}

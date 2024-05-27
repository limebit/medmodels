"""Metrics for comparing vectors in the context of matching classes."""

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


METRICS = {
    "absolute": absolute_metric,
    "exact": exact_metric,
}

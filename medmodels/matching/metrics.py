from typing import Tuple

import numpy as np


def absolute_metric(*vectors: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculates the Euclidean distance (L1 norm) between two vectors, providing a measure
    of the absolute difference between them. This distance is the sum of the absolute
    differences between each corresponding pair of elements in the two vectors.

    Args:
        vectors (Tuple[np.ndarray, np.ndarray]): Two numpy arrays to be compared.

    Returns:
        float: The Euclidean distance between the two vectors.

    The calculation is based on the formula:
        $$
        D(x, y) = ||x - y||_1 = \\sum_{i=1}^n |x_i - y_i| for x, y \\in \\mathbb{R}^n
        $$
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
    \\begin{cases}
    0      & \\text{if } x = y, \\\\
    \\infty & \\text{otherwise}.
    \\end{cases}
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


METRICS = {
    "absolute": absolute_metric,
    "exact": exact_metric,
}

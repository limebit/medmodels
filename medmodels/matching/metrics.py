import numpy as np
import math


def absolute_metric(*vectors):

    """
    Returns the euclidean distance (first norm) between two vectors.

    .. math::
        D(x, y) = ||x - y||_1 = \\sum_{i=1}^n |x_i - y_i| for x, y \\in \\mathbb{R}^n

    @param vectors: two numpy arrays to be compared
    @return: the euclidean metric
    """

    diff = vectors[0] - vectors[1]

    return sum(np.abs(diff))


def exact_metric(*vectors):

    """
    Returns the exact metric for matching. Better not to use with continuous covariates.

    .. math::
        D(x, y) = \begin{cases}
                  0      & \text{if x = y}, \\
                  \\infty & \text{otherwise}
                  \\end{cases}

    @param vectors: two numpy arrays to be compared
    @return: 0 if the vectors are equal, infinity if not
    """
    v1, v2 = vectors[0], vectors[1]
    if np.array_equal(v1, v2):
        return 0
    else:
        return np.inf


def mahalanobis_metric(*vectors, inv_cov):

    """
    Returns mahalanobis metric for matching. Works better with continuous covariates.

    .. math::
        D(x, y) = \\sqrt{(x-y)^T S^{-1} (x-y)} where S is the covariate matrix of
        the whole distribution

    @param vectors: two numpy arrays (vectors from the same distribution) to be compared
    @param inv_cov: the inverse of the covariance matrix of the whole distribution (data
                    set)
    @return: mahalanobis distance between the vectors

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
    return math.sqrt(right)


METRICS = {
    "absolute": absolute_metric,
    "exact": exact_metric,
    "mahalanobis": mahalanobis_metric,
}

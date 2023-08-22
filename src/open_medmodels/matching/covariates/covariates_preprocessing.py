import numpy as np
import pandas as pd


def covariate_coarsen(covariate, n_bins=10):

    """
    Bins a continuous variable.

    :param covariate: the variable to be binned;
    :param n_bins: number of clusters;
    :return: discrete array.

    Example:
    x = [1, 5, 10, 14, 15], n_bins = 3
    bins = [0.99, 5.66333333, 10.33666667, 15.01]
           (0.99 <= x < 5.66333333,
            5.66333333 <= x < 10.33666667,
            10.33666667 <= x < 15.01])  <- the strict inequality doesn't matter because
                                           the last border is higher than the max value
                                           of the covariate
    return [1, 1, 2, 3, 3]

    Also works if all the values are equal (because a small noise is added).
    """

    bins = np.linspace(min(covariate) - 0.01, max(covariate) + 0.01, n_bins + 1)

    return np.digitize(covariate, bins)


def covariate_add_noise(covariate, n_digits=2):

    """
    Adds noise after n_digits decimal places to a discrete variable to consider it as
    continuous. Needed mostly for examples and tests.

    :param covariate: the discrete variable to add the noise to;
    :param n_digits: the number of digit places (e.g. 2 means the noise btw 0 and 0.01),
                     can also be negative;
    :return: pandas Series containing the new covariate.
    """

    noise = pd.Series(np.random.rand(covariate.shape[0])) * 10 ** (-n_digits)

    return covariate.add(noise)

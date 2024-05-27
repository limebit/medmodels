from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def covariate_coarsen(
    covariate: NDArray[Union[np.float64, np.int64]], n_bins: int = 10
) -> NDArray[Union[np.float64, np.int64]]:
    """
    Bins a continuous variable into discrete intervals. This method divides the range of
    `covariate` into `n_bins` equal-width bins and assigns each value to a bin
    represented by a discrete integer. It ensures functionality even when all covariate
    values are equal by adding a small noise.

    Args:
        covariate (NDArray[Union[np.float64, np.int64]]): The continuous variable to be
            binned.
        n_bins (int, optional): The number of bins to divide the covariate into.
            Defaults to 10.

    Returns:
        NDArray[Union[np.float64, np.int64]]: An array of discrete integers representing
            the bin assignments for each entry in `covariate`.

    Example:
        For `covariate` = [1, 5, 10, 14, 15] and `n_bins` = 3, the function might
        return [1, 1, 2, 3, 3], indicating the bin assignment for each value in
        `covariate`.
    """

    bins = np.linspace(min(covariate) - 0.01, max(covariate) + 0.01, n_bins + 1)

    return np.digitize(covariate, bins)


def covariate_add_noise(
    covariate: pd.Series[Union[int, float]], n_digits: int = 2
) -> pd.Series[float]:
    """
    Adds noise after a specified number of decimal places to a discrete variable,
    transforming it into a continuous variable. This is particularly useful for
    simulations, examples, and tests, allowing discrete variables to be used in contexts
    requiring continuous variables.

    Args:
        covariate (pd.Series[Union[int, float]]): The discrete variable to be
            transformed.
        n_digits (int, optional): Specifies the decimal place after which to add noise.
            A positive value adds noise with a magnitude less than 1, while a negative
            value can increase the noise magnitude. Defaults to 2, resulting in noise
            between 0 and 0.01.

    Returns:
        pd.Series[float]: A pandas Series containing the modified covariate with added
            noise.

    Example:
        If `covariate` is a Series of integers and `n_digits` is 2, the function will
        add a random noise between 0 and 0.01 to each entry, effectively making the
        variable continuous.
    """

    noise = pd.Series(np.random.rand(covariate.shape[0])) * 10 ** (-n_digits)

    return covariate.add(noise)

import unittest

import numpy as np
import pandas as pd

from medmodels.matching.covariates import covariates_preprocessing as cp


class TestCovariatesPreprocessing(unittest.TestCase):
    def test_covariate_coarsen(self):
        result = cp.covariate_coarsen(np.array([1, 5, 10, 14, 15]), n_bins=3)
        self.assertTrue(np.all(result == [1, 1, 2, 3, 3]))

        result = cp.covariate_coarsen(np.array([1, 1]))
        self.assertTrue(np.all(result == [6, 6]))

    def test_covariate_add_noise(self):
        #  Check if the Series are not equal:
        result = cp.covariate_add_noise(pd.Series([1, 2]), 1)
        self.assertTrue(not result.equals(pd.Series([1, 2])))

        # Check if series are equal till the certain precision:
        expected = pd.Series([1, 2]).astype(float)
        pd.testing.assert_series_equal(result.round(), expected)
        # here maybe a better test, maybe with round.
        # But 0.99999 + 0.001 = 1.00099 so round will give an error.


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestCovariatesPreprocessing)
    unittest.TextTestRunner(verbosity=2).run(run_test)

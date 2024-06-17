import unittest

import numpy as np
import polars as pl

from medmodels.matching.algorithms import classic_distance_models as cdm


class TestClassicDistanceModels(unittest.TestCase):
    def test_nearest_neighbor(self):
        ###########################################
        # 1D example
        c_set = pl.DataFrame({"a": [1, 5, 1, 3]})
        t_set = pl.DataFrame({"a": [1, 4]})

        # absolute metric
        expected_abs = pl.DataFrame({"a": [1.0, 5.0]})
        result_abs = cdm.nearest_neighbor(t_set, c_set, "absolute")
        self.assertTrue(result_abs.equals(expected_abs))

        ###########################################
        # 3D example with covariates
        cols = ["a", "b", "c"]
        array = np.array([[1, 3, 5], [5, 2, 1], [1, 4, 10]])
        c_set = pl.DataFrame(array, schema=cols)
        t_set = pl.DataFrame([[1, 4, 2]], schema=cols)
        covs = ["a", "c"]

        # absolute metric
        expected_abs = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols)
        result_abs = cdm.nearest_neighbor(t_set, c_set, "absolute", covariates=covs)
        self.assertTrue(result_abs.equals(expected_abs))

        # mahalanobis metric
        expected_mah = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols)
        result_mah = cdm.nearest_neighbor(t_set, c_set, "mahalanobis", covariates=covs)
        self.assertTrue(result_mah.equals(expected_mah))

        expected_abs_2nn = pl.DataFrame(
            [[1.0, 3.0, 5.0], [1.0, 4.0, 10.0]], schema=cols
        )
        result_abs_2nn = cdm.nearest_neighbor(
            t_set, c_set, "absolute", covariates=covs, number_of_neighbors=2
        )
        self.assertTrue(result_abs_2nn.equals(expected_abs_2nn))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestClassicDistanceModels)
    unittest.TextTestRunner(verbosity=2).run(run_test)

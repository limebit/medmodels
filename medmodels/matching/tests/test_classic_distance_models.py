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

        expected_result = pl.DataFrame({"a": [1.0, 5.0]})
        result = cdm.nearest_neighbor(t_set, c_set)
        self.assertTrue(result.equals(expected_result))

        ###########################################
        # 3D example with covariates
        cols = ["a", "b", "c"]
        array = np.array([[1, 3, 5], [5, 2, 1], [1, 4, 10]])
        c_set = pl.DataFrame(array, schema=cols)
        t_set = pl.DataFrame([[1, 4, 2]], schema=cols, orient="row")
        covs = ["a", "c"]

        expected_result = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols, orient="row")
        result = cdm.nearest_neighbor(t_set, c_set, covariates=covs)
        self.assertTrue(result.equals(expected_result))

        # 2 nearest neighbors
        expected_abs_2nn = pl.DataFrame(
            [[1.0, 3.0, 5.0], [1.0, 4.0, 10.0]], schema=cols, orient="row"
        )
        result_abs_2nn = cdm.nearest_neighbor(
            t_set, c_set, covariates=covs, number_of_neighbors=2
        )
        self.assertTrue(result_abs_2nn.equals(expected_abs_2nn))

    def test_nearest_neighbor_value_error(self):
        # Test case for checking the ValueError when all control units have been matched
        c_set = pl.DataFrame({"a": [1, 2]})
        t_set = pl.DataFrame({"a": [1, 2, 3]})

        with self.assertRaises(ValueError) as context:
            cdm.nearest_neighbor(t_set, c_set, number_of_neighbors=2)

        self.assertEqual(
            str(context.exception),
            "The treated set is too large for the given number of neighbors.",
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestClassicDistanceModels)
    unittest.TextTestRunner(verbosity=2).run(run_test)

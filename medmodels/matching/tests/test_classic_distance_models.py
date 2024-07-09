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

        # minkowski metric
        expected_abs = pl.DataFrame({"a": [1.0, 5.0]})
        result_abs = cdm.nearest_neighbor(t_set, c_set, "minkowski")
        self.assertTrue(result_abs.equals(expected_abs))

        ###########################################
        # 3D example with covariates
        cols = ["a", "b", "c"]
        array = np.array([[1, 3, 5], [5, 2, 1], [1, 4, 10]])
        c_set = pl.DataFrame(array, schema=cols)
        t_set = pl.DataFrame([[1, 4, 2]], schema=cols, orient="row")
        covs = ["a", "c"]

        # minkowksi metric
        expected_mink = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols, orient="row")
        result_mink = cdm.nearest_neighbor(
            t_set, c_set, metric="minkowski", covariates=covs
        )
        self.assertTrue(result_mink.equals(expected_mink))

        # euclidean metric
        expected_euc = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols, orient="row")
        result_euc = cdm.nearest_neighbor(t_set, c_set, "euclidean", covariates=covs)
        self.assertTrue(result_euc.equals(expected_euc))

        # ball_tree algorithm
        expected_ball_tree = pl.DataFrame([[1.0, 3.0, 5.0]], schema=cols, orient="row")
        result_ball_tree = cdm.nearest_neighbor(
            t_set, c_set, algorithm="ball_tree", covariates=covs
        )
        self.assertTrue(result_ball_tree.equals(expected_ball_tree))

        # 2 nearest neighbors
        expected_abs_2nn = pl.DataFrame(
            [[1.0, 3.0, 5.0], [5.0, 2.0, 1.0]], schema=cols, orient="row"
        )
        result_abs_2nn = cdm.nearest_neighbor(
            t_set, c_set, "minkowski", covariates=covs, number_of_neighbors=2
        )
        self.assertTrue(result_abs_2nn.equals(expected_abs_2nn))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestClassicDistanceModels)
    unittest.TextTestRunner(verbosity=2).run(run_test)

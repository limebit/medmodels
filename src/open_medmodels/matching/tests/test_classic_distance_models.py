import unittest
from open_medmodels.matching.algorithms import classic_distance_models as cdm
import numpy as np
import pandas as pd


class TestClassicDistanceModels(unittest.TestCase):
    def test_nearest_neighbor(self):

        ###########################################
        # 1D example
        c_set = pd.DataFrame(np.array([[1], [5], [1], [3]]), columns=["a"])
        t_set = pd.DataFrame(np.array([[1], [4]]), columns=["a"])

        # absolute metric
        expected_abs = pd.DataFrame(np.array([[1.0], [5.0]]), columns=["a"])
        result_abs = cdm.nearest_neighbor(t_set, c_set, "absolute")
        pd.testing.assert_frame_equal(result_abs, expected_abs)

        # mahalanobis metric
        expected_mah = pd.DataFrame(np.array([[1.0], [5.0]]), columns=["a"])
        result_mah = cdm.nearest_neighbor(t_set, c_set, "mahalanobis")
        pd.testing.assert_frame_equal(result_mah, expected_mah)

        ###########################################
        # 3D example with covariates
        cols = ["a", "b", "c"]
        array = np.array([[1, 3, 5], [5, 2, 1], [1, 4, 10]])
        c_set = pd.DataFrame(array, columns=cols)
        t_set = pd.DataFrame(np.array([[1, 4, 2]]), columns=cols)
        covs = ["a", "c"]

        # absolute metric
        expected_abs = pd.DataFrame(np.array([[1.0, 3.0, 5.0]]), columns=cols)
        result_abs = cdm.nearest_neighbor(t_set, c_set, "absolute", covariates=covs)
        pd.testing.assert_frame_equal(result_abs, expected_abs)

        # mahalanobis metric
        expected_mah = pd.DataFrame(np.array([[1.0, 3.0, 5.0]]), columns=cols)
        result_mah = cdm.nearest_neighbor(t_set, c_set, "mahalanobis", covariates=covs)
        pd.testing.assert_frame_equal(result_mah, expected_mah)


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestClassicDistanceModels)
    unittest.TextTestRunner(verbosity=2).run(run_test)

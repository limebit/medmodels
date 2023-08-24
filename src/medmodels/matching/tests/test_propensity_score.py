import unittest
from medmodels.matching.algorithms import propensity_score as ps
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class TestPropensityScore(unittest.TestCase):
    def test_calculate_propensity(self):

        x, y = load_iris(return_X_y=True)

        # Set random state by each propensity estimator:
        hyperparam = {"random_state": 1}
        hyperparam_logit = {"random_state": 1, "max_iter": 200}

        # Logistic Regression metric:
        result_1, result_2 = ps.calculate_propensity(
            x, y, [x[0, :]], [x[1, :]], hyperparam=hyperparam_logit
        )
        self.assertAlmostEqual(result_1[0], 1.43580537e-08, places=9)
        self.assertAlmostEqual(result_2[0], 3.00353141e-08, places=9)

        # Decision Tree Classifier metric:
        result_1, result_2 = ps.calculate_propensity(
            x, y, [x[0, :]], [x[1, :]], metric="dec_tree", hyperparam=hyperparam
        )
        self.assertAlmostEqual(result_1[0], 0, places=2)
        self.assertAlmostEqual(result_2[0], 0, places=2)

        # Random Forest Classifier metric:
        result_1, result_2 = ps.calculate_propensity(
            x, y, [x[0, :]], [x[1, :]], metric="forest", hyperparam=hyperparam
        )
        self.assertAlmostEqual(result_1[0], 0, places=2)
        self.assertAlmostEqual(result_2[0], 0, places=2)

    def test_run_propensity_score(self):

        # Set random state by each propensity estimator:
        hyperparam = {"random_state": 1}
        hyperparam_logit = {"random_state": 1, "max_iter": 200}

        ###########################################
        # 1D example
        control_set = pd.DataFrame(np.array([[1], [5], [1], [3]]), columns=["a"])
        treated_set = pd.DataFrame(np.array([[1], [4]]), columns=["a"])

        # logit metric
        expected_logit = pd.DataFrame(np.array([[1.0], [3.0]]), columns=["a"])
        result_logit = ps.run_propensity_score(
            treated_set, control_set, hyperparam=hyperparam_logit
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)

        # dec_tree metric
        expected_logit = pd.DataFrame(np.array([[1.0], [1.0]]), columns=["a"])
        result_logit = ps.run_propensity_score(
            treated_set, control_set, model="dec_tree", hyperparam=hyperparam
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)

        # forest metric
        expected_logit = pd.DataFrame(np.array([[1.0], [1.0]]), columns=["a"])
        result_logit = ps.run_propensity_score(
            treated_set, control_set, model="forest", hyperparam=hyperparam
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)

        ###########################################
        # 3D example with covariates
        cols = ["a", "b", "c"]
        array = np.array([[1, 3, 5], [5, 2, 1], [1, 4, 10]])
        control_set = pd.DataFrame(array, columns=cols)
        treated_set = pd.DataFrame(np.array([[1, 4, 2]]), columns=cols)
        covs = ["a", "c"]

        # logit metric
        expected_logit = pd.DataFrame(np.array([[1.0, 3.0, 5.0]]), columns=cols)
        result_logit = ps.run_propensity_score(
            treated_set, control_set, covariates=covs, hyperparam=hyperparam_logit
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)

        # dec_tree metric
        expected_logit = pd.DataFrame(np.array([[1.0, 3.0, 5.0]]), columns=cols)
        result_logit = ps.run_propensity_score(
            treated_set,
            control_set,
            model="dec_tree",
            covariates=covs,
            hyperparam=hyperparam,
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)

        # forest metric
        expected_logit = pd.DataFrame(np.array([[1.0, 3.0, 5.0]]), columns=cols)
        result_logit = ps.run_propensity_score(
            treated_set,
            control_set,
            model="forest",
            covariates=covs,
            hyperparam=hyperparam,
        )
        pd.testing.assert_frame_equal(result_logit, expected_logit)


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestPropensityScore)
    unittest.TextTestRunner(verbosity=2).run(run_test)

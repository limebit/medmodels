import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from medmodels.matching import evaluation


class TestEvaluation(unittest.TestCase):
    """Test class for the evaluation module."""

    def test_relative_diff(self):
        """Test the calculate_relative_diff function."""
        row = pd.Series({"control_mean": 2.0, "treated_mean": 3.0})
        expected = 50.0  # (3 - 2) / 2 * 100 = 50%
        result = evaluation.calculate_relative_diff(row)
        self.assertAlmostEqual(result, expected)

    def test_relative_diff_in_means(self):
        """Test the relative_diff_in_means function."""
        df_control = pd.DataFrame(np.array([[1, 3], [3, -3]]), columns=["a", "b"])
        df_treated = pd.DataFrame(np.array([[3, 4], [4, -10]]), columns=["a", "b"])

        s = pd.Series(["control_mean", "treated_mean", "Diff (in %)"])
        expected = pd.DataFrame(
            np.array([[2, 0], [3.5, -3], [75, 300]]), columns=["a", "b"]
        )
        expected = expected.set_index(s)

        result = evaluation.relative_diff_in_means(df_control, df_treated)

        assert_frame_equal(result, expected)

    def test_average_value_over_features(self):
        """Test the average_value_over_features function."""
        s = pd.Series(["control_mean", "treated_mean", "Diff (in %)"])
        df = pd.DataFrame(np.array([[2, 2], [3.5, 7], [75, 250]]), columns=["a", "b"])
        df = df.set_index(s)

        self.assertEqual(evaluation.average_value_over_features(df), 162.5)

    def test_aard(self):
        """Test the absolute average relative difference function."""
        cols = ["a", "b", "c"]
        df_control = pd.DataFrame(np.array([[1, 3, 5], [3, 1, 7]]), columns=cols)
        df_treated = pd.DataFrame(np.array([[3, 4, 10], [4, 10, 5]]), columns=cols)

        covariates = ["a", "c"]
        result_mean, _ = evaluation.average_abs_relative_diff(
            df_control, df_treated, covariates
        )

        self.assertEqual(result_mean, 50.0)


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestEvaluation)
    unittest.TextTestRunner(verbosity=2).run(run_test)

import unittest
import numpy as np
import pandas as pd
import open_medmodels.treatment_effect_estimation.continuous_estimators as ce


class TestContinuousEstimators(unittest.TestCase):
    def test_average_treatment_effect(self):
        df_control = pd.DataFrame(np.array([[1, 3], [3, 1]]), columns=["a", "b"])
        df_treated = pd.DataFrame(np.array([[3, 4], [4, -10]]), columns=["a", "b"])
        outcome_variable = "b"

        result = ce.average_treatment_effect(df_treated, df_control, outcome_variable)

        self.assertEqual(result, -5.0)

    def test_cohen_d(self):
        x = np.array([2, 4, 7, 3, 7, 35, 8, 9])
        y = x * 2
        df_treated = pd.DataFrame(x, columns=["a"])
        df_control = pd.DataFrame(y, columns=["a"])
        outcome_variable = "a"

        result = ce.cohen_d(df_treated, df_control, outcome_variable)
        self.assertEqual(round(result, 4), -0.5568)

        result = ce.cohen_d(
            df_treated, df_control, outcome_variable, add_correction=True
        )
        self.assertEqual(round(result, 4), -0.4193)


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestContinuousEstimators)
    unittest.TextTestRunner(verbosity=2).run(run_test)

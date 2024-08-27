import unittest

import numpy as np

from medmodels.treatment_effect.matching.covariates import metrics


class TestMetrics(unittest.TestCase):
    def test_absolute_metric(self) -> None:
        assert metrics.absolute_metric(np.array([-2]), np.array([-1])) == 1
        assert metrics.absolute_metric(np.array([2, -1]), np.array([1, 3])) == 5

    def test_exact_metric(self) -> None:
        assert metrics.exact_metric(np.array([-2]), np.array([-2])) == 0
        assert metrics.exact_metric(np.array([-2]), np.array([-1])) == np.inf
        assert metrics.exact_metric(np.array([2, -1]), np.array([2, -1])) == 0
        assert metrics.exact_metric(np.array([2, -1]), np.array([2, 1])) == np.inf

    def test_mahalanobis_metric(self) -> None:
        data = np.array(
            [[64, 66, 68, 69, 73], [580, 570, 590, 660, 600], [29, 33, 37, 46, 55]]
        )
        inv_cov = np.linalg.inv(np.cov(data))
        a1, a2 = np.array([68, 600, 40]), np.array([66, 640, 44])
        result = metrics.mahalanobis_metric(a1, a2, inv_cov=inv_cov)
        self.assertAlmostEqual(result, 5.33, 2)

        data = np.array([[-2.1, -1, 4.3]])
        inv_cov = 1 / np.cov(data)
        a1, a2 = np.array([1]), np.array([2])
        result = metrics.mahalanobis_metric(a1, a2, inv_cov=inv_cov)
        self.assertAlmostEqual(result, 0.29, 2)


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMetrics)
    unittest.TextTestRunner(verbosity=2).run(run_test)

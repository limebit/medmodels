import unittest
from medmodels.matching import metrics
import numpy as np


class TestMetrics(unittest.TestCase):
    def test_absolute_metric(self):

        self.assertEqual(metrics.absolute_metric(np.array([-2]), np.array([-1])), 1)
        self.assertEqual(
            metrics.absolute_metric(np.array([2, -1]), np.array([1, 3])), 5
        )

    def test_exact_metric(self):

        self.assertEqual(metrics.exact_metric(np.array([-2]), np.array([-2])), 0)
        self.assertEqual(metrics.exact_metric(np.array([-2]), np.array([-1])), np.inf)
        self.assertEqual(metrics.exact_metric(np.array([2, -1]), np.array([2, -1])), 0)
        self.assertEqual(
            metrics.exact_metric(np.array([2, -1]), np.array([2, 1])), np.inf
        )

    def test_mahalanobis_metric(self):

        data = np.array(
            [[64, 66, 68, 69, 73], [580, 570, 590, 660, 600], [29, 33, 37, 46, 55]]
        )
        inv_cov = np.linalg.inv(np.cov(data))
        a1, a2 = np.array([68, 600, 40]), np.array([66, 640, 44])
        result = round(metrics.mahalanobis_metric(a1, a2, inv_cov=inv_cov), 2)
        self.assertEqual(result, 5.33)

        data = np.array([[-2.1, -1, 4.3]])
        inv_cov = np.array([1 / np.cov(data)])
        a1, a2 = np.array([1]), np.array([2])
        result = round(metrics.mahalanobis_metric(a1, a2, inv_cov=inv_cov), 2)
        self.assertEqual(result, 0.29)


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMetrics)
    unittest.TextTestRunner(verbosity=2).run(run_test)

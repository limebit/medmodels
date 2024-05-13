import unittest

import numpy as np

from medmodels.matching import metrics


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


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMetrics)
    unittest.TextTestRunner(verbosity=2).run(run_test)

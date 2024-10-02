import unittest

import torch

from medmodels.data_synthesis.mtgan.model.masks import find_sequence_mask


class TestFindSequenceMask(unittest.TestCase):
    def test_find_sequence_mask(self):
        """Test the find_sequence_mask function."""
        number_of_windows_per_patient = torch.tensor([1, 3, 2, 2])
        maximum_number_of_windows = 3

        expected_mask = torch.tensor(
            [
                [[True], [False], [False]],
                [[True], [True], [True]],
                [[True], [True], [False]],
                [[True], [True], [False]],
            ]
        ).squeeze()

        result_mask = find_sequence_mask(
            number_of_windows_per_patient, maximum_number_of_windows
        )

        self.assertTrue(torch.all(result_mask.eq(expected_mask)))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestFindSequenceMask)
    unittest.TextTestRunner(verbosity=2).run(run_test)

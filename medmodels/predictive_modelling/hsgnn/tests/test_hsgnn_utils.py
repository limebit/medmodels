import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from medmodels.predictive_modeling.hsgnn.hsgnn_utils import load_hyperparameters


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.mock_hyperparams = {
            "preprocessing": {
                "param1": 5,
                "param2": 64,
                "param3": 750,
            },
            "model": {
                "param1": 128,
                "param2": 64,
            },
        }
        self.expected_hyperparams_preprocessing = self.mock_hyperparams["preprocessing"]
        self.expected_hyperparams_model = self.mock_hyperparams["model"]
        self.mock_hyperparams_path = Path(
            "src/medmodels/predictive_modeling/hsgnn/hyperparameters.json"
        )

        self.mock_medrecord_path = Path()

    def test_load_hyperparameters(self):
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.mock_hyperparams))
        ):
            result = load_hyperparameters(
                hyperparams_path=self.mock_hyperparams_path,
                required_hyperparams={"param1", "param2"},
                hyperparams_type="model",
            )
            self.assertEqual(result, self.expected_hyperparams_model)

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.mock_hyperparams))
        ):
            with self.assertRaises(KeyError):
                load_hyperparameters(
                    hyperparams_path=self.mock_hyperparams_path,
                    required_hyperparams={"param1", "param2", "param3"},
                    hyperparams_type="model",
                )

        with self.assertRaises(FileNotFoundError):
            load_hyperparameters(
                hyperparams_path=Path("nonexistent_path.json"),
                required_hyperparams={"param1", "param2"},
                hyperparams_type="model",
            )

        with self.assertRaises(AssertionError):
            load_hyperparameters(
                hyperparams_path=self.mock_hyperparams_path,
                required_hyperparams={"param1", "param2"},
                hyperparams_type="invalid_type",
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(run_test)

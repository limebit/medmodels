import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from medmodels.predictive_modeling.hsgnn.model_preprocessing import ModelPreprocessor
from medmodels.predictive_modeling.hsgnn.tests.mock_medrecord import (
    MockMedRecord,
    MockSubgraphs,
)


class TestModelPreprocessor(unittest.TestCase):
    def setUp(self):
        self.medrecord = MockMedRecord().create_mock_medrecord()
        self.subgraphs = MockSubgraphs().create_mock_subgraphs()
        self.model_preprocessor = ModelPreprocessor(
            self.medrecord,
            data_path="src/medmodels/predictive_modeling/hsgnn/tests/mock_medrecord.py",
            hyperparams_path=Path(
                "/Users/anastasiia.tiurina/Desktop/projects/medmodels/src/medmodels/predictive_modeling/hsgnn/hyperparameters.json"
            ),
        )
        mock_embeddings = {
            "P-1": np.random.rand(64),
            "P-2": np.random.rand(64),
            "P-3": np.random.rand(64),
            "1": np.random.rand(64),
            "D-2": np.random.rand(64),
            "prescriptions_1": np.random.rand(64),
            "M-2": np.random.rand(64),
        }
        self.mock_embeddings_df = pd.DataFrame(mock_embeddings).T

        self.expected_pyg_data = Data(
            x=torch.tensor(self.mock_embeddings_df.values),
            edge_index=torch.tensor(
                [[0, 0, 0, 0, 1, 1], [3, 4, 5, 6, 4, 6]], dtype=torch.int64
            ),
            edge_attr=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float16),
        )

    def test_correct_node_name(self):
        # test the Assertion for the prefix name
        with self.assertRaises(AssertionError):
            self.model_preprocessor._correct_node_name("Q-1")

        # test if the node originally saved in graph without dimension name
        self.assertEqual(self.model_preprocessor._correct_node_name("D-1"), "1")

        # test if the node was assigned the dimension name
        self.assertEqual(
            self.model_preprocessor._correct_node_name("M-1"), "prescriptions_1"
        )

        # test the Error if the node not in the graph
        with self.assertRaises(KeyError):
            self.model_preprocessor._correct_node_name("M-13")

    def test_remove_node(self):
        expected_node_list = ["P-1", "P-2"]
        self.model_preprocessor._remove_node("P-3")

        # test the Assertion if the node not in the graph
        with self.assertRaises(AssertionError):
            self.model_preprocessor._remove_node("Q-1")

        # test if the node was removed from dimension list
        self.assertEqual(expected_node_list, self.medrecord.dimension("patients"))

        # test if the node was removed from _node_mapping
        self.assertEqual(expected_node_list, self.medrecord._node_mapping["patients"])

        # test if the node was removed from the graph
        expected_graph_nodes = ["P-1", "P-2", "1", "D-2", "prescriptions_1", "M-2"]
        self.assertEqual(expected_graph_nodes, list(self.medrecord.G.nodes()))

    def test_compute_embeddings(self):
        computed_embeddings = self.model_preprocessor.compute_embeddings()
        self.assertIsInstance(computed_embeddings, pd.DataFrame)
        self.assertEqual(computed_embeddings.shape, (7, 64))

    # TODO: what is the best way to test this method?
    # def test_compute_embeddings_mce(self):
    #     computed_embeddings = self.model_preprocessor.compute_embeddings_mce()
    #     self.assertIsInstance(computed_embeddings, pd.DataFrame)
    #     self.assertEqual(computed_embeddings.shape, (7, 64))

    @patch("pandas.DataFrame.to_pickle")
    def test_save_embeddings(self, mock_to_pickle):
        self.model_preprocessor.save_embeddings(
            self.mock_embeddings_df,
            Path("src/medmodels/predictive_modeling/hsgnn/tests/embeddings.pkl"),
        )
        mock_to_pickle.assert_called_once_with(
            Path("src/medmodels/predictive_modeling/hsgnn/tests/embeddings.pkl"),
        )

    def test_to_PyG_Data(self):
        computed_pyg_data = self.model_preprocessor.to_PyG_Data(
            self.mock_embeddings_df, self.subgraphs
        )

        self.assertTrue(torch.equal(computed_pyg_data.x, self.expected_pyg_data.x))
        self.assertTrue(
            torch.equal(computed_pyg_data.edge_index, self.expected_pyg_data.edge_index)
        )
        self.assertTrue(
            torch.equal(computed_pyg_data.edge_attr, self.expected_pyg_data.edge_attr)
        )

    def test_node_types_mapping(self):
        expected_node_types_tensor_options = [
            torch.tensor([0, 0, 0, 1, 1, 2, 2]),
            torch.tensor([0, 0, 0, 2, 2, 1, 1]),
            torch.tensor([1, 1, 1, 0, 0, 2, 2]),
            torch.tensor([1, 1, 1, 2, 2, 0, 0]),
            torch.tensor([2, 2, 2, 0, 0, 1, 1]),
            torch.tensor([2, 2, 2, 1, 1, 0, 0]),
        ]
        computed_node_types_tensor = self.model_preprocessor.node_types_mapping()
        matched = False
        for expected_node_types_tensor in expected_node_types_tensor_options:
            if torch.equal(computed_node_types_tensor, expected_node_types_tensor):
                matched = True
                break

        self.assertTrue(matched)

    def test_prepare_data_for_model(self):
        (
            computed_pyg_data,
            computed_subgraphs,
            computed_node_types_tensor,
        ) = self.model_preprocessor.prepare_data_for_model()

        self.assertEqual(computed_pyg_data.x.shape, self.expected_pyg_data.x.shape)
        self.assertTrue(
            torch.equal(computed_pyg_data.edge_index, self.expected_pyg_data.edge_index)
        )
        self.assertTrue(
            torch.equal(computed_pyg_data.edge_attr, self.expected_pyg_data.edge_attr)
        )

        for i in range(len(computed_subgraphs)):
            self.assertTrue(
                torch.equal(
                    computed_subgraphs[i].to_dense(), self.subgraphs[i].to_dense()
                )
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestModelPreprocessor)
    unittest.TextTestRunner(verbosity=2).run(run_test)

import unittest

import torch

from medmodels.predictive_modeling.hsgnn.hsgnn_model import (
    SimplyWeightedSum,
)
from medmodels.predictive_modeling.hsgnn.tests.mock_medrecord import (
    MockMedRecord,
    MockSubgraphs,
)


class TestSimplyWeightedSum(unittest.TestCase):
    def setUp(self):
        self.medrecord = MockMedRecord().create_mock_medrecord()
        self.subgraphs = MockSubgraphs().create_mock_subgraphs()
        self.weighted_sum = SimplyWeightedSum(number_of_subgraphs=5)

    def test_initialize_norm_weights(self):
        computed_weights = self.weighted_sum.weights

        self.assertEqual(len(computed_weights), 5)

        # All weights have to sum up to 1
        sum_of_computed_weights = torch.sum(
            torch.cat([weight for weight in computed_weights])
        )
        self.assertAlmostEqual(sum_of_computed_weights.item(), 1.0, places=5)

    def test_forward(self):
        computed_fused_matrix = self.weighted_sum.forward(self.subgraphs)

        self.assertEqual(computed_fused_matrix.shape, (7, 7))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestSimplyWeightedSum)
    unittest.TextTestRunner(verbosity=2).run(run_test)

# class TestHSGNNModel(unittest.TestCase):
#     def setUp(self):
#         self.medrecord = MockMedRecord().create_mock_medrecord()
#         self.subgraphs = MockSubgraphs().create_mock_subgraphs()
#         self.weighted_sum = SimplyWeightedSum(number_of_subgraphs=5)
#         self.model = HSGNNModel(64, 32, 16, 5, torch.tensor([0, 0, 0, 1, 1, 2, 2]))
#         mock_embeddings = {
#             "P-1": np.random.rand(64),
#             "P-2": np.random.rand(64),
#             "P-3": np.random.rand(64),
#             "1": np.random.rand(64),
#             "D-2": np.random.rand(64),
#             "prescriptions_1": np.random.rand(64),
#             "M-2": np.random.rand(64),
#         }
#         self.mock_embeddings_df = pd.DataFrame(mock_embeddings).T

#         self.expected_pyg_data = Data(
#             x=torch.tensor(self.mock_embeddings_df.values),
#             edge_index=torch.tensor(
#                 [[0, 0, 0, 0, 1, 1], [3, 4, 5, 6, 4, 6]], dtype=torch.int64
#             ),
#             edge_attr=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#         )

#     def test_all_parameters_updates(self):
#         node_embeddings, fused_matrix = self.model(
#             self.expected_pyg_data.x, self.expected_pyg_data.edge_index, self.subgraphs
#         )
#         self.assertEqual(node_embeddings.size(), (7, 16))
#         self.assertEqual(fused_matrix.size(), (7, 7))
#         print(fused_matrix)
#         print(node_embeddings)


# run_test = unittest.TestLoader().loadTestsFromTestCase(TestHSGNNModel)
# unittest.TextTestRunner(verbosity=2).run(run_test)

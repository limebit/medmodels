import unittest

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix

from medmodels.predictive_modeling.hsgnn.hsgnn_preprocessing import HSGNNPreprocessor
from medmodels.predictive_modeling.hsgnn.tests.mock_medrecord import MockMedRecord


class TestHSGNNPreprocessor(unittest.TestCase):
    def setUp(self):
        self.medrecord = MockMedRecord().create_mock_medrecord()
        self.hsgnn_preprocessor = HSGNNPreprocessor(self.medrecord)

        self.expected_rows_symmetric = torch.tensor([0, 0, 1, 1])
        self.expected_cols_symmetric = torch.tensor([0, 1, 0, 1])
        self.expected_similarity_scores_symmetric = torch.tensor([1, 2 / 3, 2 / 3, 1])
        self.expected_matrix_symmetric = torch.sparse_coo_tensor(
            torch.stack([self.expected_rows_symmetric, self.expected_cols_symmetric]),
            self.expected_similarity_scores_symmetric,
            size=(7, 7),
            dtype=torch.float16,
        ).to_dense()

        self.expected_rows_non_symmetric = torch.tensor([3, 3, 4, 4, 5, 6, 5, 6])
        self.expected_cols_non_symmetric = torch.tensor([5, 6, 5, 6, 3, 3, 4, 4])
        self.expected_similarity_scores_non_symmetric = torch.tensor(
            [1, 2 / 3, 2 / 3, 1, 1, 2 / 3, 2 / 3, 1]
        )
        self.expected_matrix_non_symmetric = torch.sparse_coo_tensor(
            torch.stack(
                [self.expected_rows_non_symmetric, self.expected_cols_non_symmetric]
            ),
            self.expected_similarity_scores_non_symmetric,
            size=(7, 7),
            dtype=torch.float16,
        ).to_dense()

    def test_create_graph_schema(self):
        computed_graph_schema = self.hsgnn_preprocessor.create_graph_schema()
        expected_graph_schema = [
            ("patients", "diagnoses"),
            ("patients", "prescriptions"),
        ]
        self.assertEqual(list(computed_graph_schema.edges), expected_graph_schema)

    def test_find_metapaths(self):
        computed_metapaths = self.hsgnn_preprocessor.find_metapaths()
        expected_metapaths = [
            ["patients", "diagnoses", "patients"],
            ["patients", "prescriptions", "patients"],
            ["diagnoses", "patients", "diagnoses"],
            ["diagnoses", "patients", "prescriptions"],
            ["prescriptions", "patients", "prescriptions"],
        ]
        self.assertEqual(computed_metapaths, expected_metapaths)

    def test_path_cout_naive(self):
        computed_path_count = self.hsgnn_preprocessor.path_count_naive(
            "P-1", "P-2", self.medrecord.dimension("diagnoses")
        )
        expected_path_count = 1
        self.assertEqual(computed_path_count, expected_path_count)

    def test_symmetric_pathsim_naive(self):
        computed_pathsim = self.hsgnn_preprocessor.symmetric_pathsim_naive(
            "P-1", "P-2", self.medrecord.dimension("diagnoses")
        )
        expected_pathsim = 2 / 3
        self.assertEqual(computed_pathsim, expected_pathsim)

    def test_symmetric_similarity_matrix_naive(self):
        # test symmetric metapaths (patients, diagnoses, patients)
        computed_matrix_symmetric = (
            self.hsgnn_preprocessor.symmetric_similarity_matrix_naive(
                meta_path=["patients", "diagnoses", "patients"]
            ).to_dense()
        )

        self.assertTrue(
            torch.equal(self.expected_matrix_symmetric, computed_matrix_symmetric)
        )

        # test non-symmetric metapaths (diagnoses, patients, prescriptions)
        computed_matrix_non_symmetric = (
            self.hsgnn_preprocessor.symmetric_similarity_matrix_naive(
                meta_path=["diagnoses", "patients", "prescriptions"]
            )
        ).to_dense()

        self.assertTrue(
            np.array_equal(
                computed_matrix_non_symmetric, self.expected_matrix_non_symmetric
            )
        )

    def test_path_count_fast(self):
        computed_path_count = self.hsgnn_preprocessor.path_count_fast(
            ["P-1", "P-2", "P-3", "1", "D-2"]
        ).toarray()
        expected_path_count = np.array(
            [
                [2, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 2],
            ]
        )
        self.assertTrue(np.array_equal(computed_path_count, expected_path_count))

    def test_symmetric_pathsim_fast(self):
        path_count_matrix = csr_matrix(
            np.array(
                [
                    [2, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 2],
                ]
            )
        )
        computed_pathsim = self.hsgnn_preprocessor.symmetric_pathsim_fast(
            path_count_matrix=path_count_matrix,
            nodes_range_first_dimension=range(0, 3),
            nodes_range_last_dimension=range(0, 3),
        ).toarray()
        expected_pathsim = np.array([[1, 2 / 3, 0], [2 / 3, 1, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(computed_pathsim, expected_pathsim))

    def test_symmetric_similarity_matrix_fast(self):
        # test symmetric metapaths (patients, diagnoses, patients)
        computed_matrix_symmetric = (
            self.hsgnn_preprocessor.symmetric_similarity_matrix_fast(
                meta_path=["patients", "diagnoses", "patients"]
            ).to_dense()
        )

        self.assertTrue(
            np.array_equal(computed_matrix_symmetric, self.expected_matrix_symmetric)
        )

        # test non-symmetric metapaths (diagnoses, patients, prescriptions)
        computed_matrix_non_symmetric = (
            self.hsgnn_preprocessor.symmetric_similarity_matrix_fast(
                meta_path=["diagnoses", "patients", "prescriptions"]
            ).to_dense()
        )

        self.assertTrue(
            np.array_equal(
                computed_matrix_non_symmetric, self.expected_matrix_non_symmetric
            )
        )

    def test_compute_matrix_indices(self):
        # test symmetric metapaths (patients, diagnoses, patients)
        similarity_scores_symmetric = coo_matrix(
            np.array([[1, 2 / 3, 0], [2 / 3, 1, 0], [0, 0, 0]])
        )
        (
            computed_rows_symmetric,
            computed_columns_symmetric,
            computed_similarity_scores_symmetric,
        ) = self.hsgnn_preprocessor.compute_matrix_indices(
            meta_path=["patients", "diagnoses", "patients"],
            similarity_scores=similarity_scores_symmetric,
        )
        self.assertTrue(
            np.array_equal(computed_rows_symmetric, self.expected_rows_symmetric)
        )
        self.assertTrue(
            np.array_equal(computed_columns_symmetric, self.expected_cols_symmetric)
        )
        # because we change precision to float16
        epsilon = 1e-3
        self.assertTrue(
            np.allclose(
                computed_similarity_scores_symmetric,
                self.expected_similarity_scores_symmetric,
                atol=epsilon,
            )
        )

        # test non-symmetric metapaths (diagnoses, patients, prescriptions)
        similarity_scores_non_symmetric = coo_matrix(np.array([[1, 2 / 3], [2 / 3, 1]]))
        (
            computed_rows_non_symmetric,
            computed_columns_non_symmetric,
            computed_similarity_scores_non_symmetric,
        ) = self.hsgnn_preprocessor.compute_matrix_indices(
            meta_path=["diagnoses", "patients", "prescriptions"],
            similarity_scores=similarity_scores_non_symmetric,
        )
        self.assertTrue(
            np.array_equal(
                computed_rows_non_symmetric, self.expected_rows_non_symmetric
            )
        )
        self.assertTrue(
            np.array_equal(
                computed_columns_non_symmetric, self.expected_cols_non_symmetric
            )
        )
        # because we change precision to float16
        epsilon = 1e-3
        self.assertTrue(
            np.allclose(
                computed_similarity_scores_non_symmetric,
                self.expected_similarity_scores_non_symmetric,
                atol=epsilon,
            )
        )

    def test_symmetric_similarity_matrix(self):
        # test symmetric metapaths (patients, diagnoses, patients)
        similarity_scores_symmetric = coo_matrix(
            np.array([[1, 2 / 3, 0], [2 / 3, 1, 0], [0, 0, 0]])
        )
        computed_matrix_symmetric = self.hsgnn_preprocessor.symmetric_similarity_matrix(
            meta_path=["patients", "diagnoses", "patients"],
            similarity_scores=similarity_scores_symmetric,
        ).to_dense()

        self.assertTrue(
            torch.equal(computed_matrix_symmetric, self.expected_matrix_symmetric)
        )

        # test non-symmetric metapaths (diagnoses, patients, prescriptions)
        similarity_scores_non_symmetric = coo_matrix(np.array([[1, 2 / 3], [2 / 3, 1]]))
        computed_matrix_non_symmetric = (
            self.hsgnn_preprocessor.symmetric_similarity_matrix(
                meta_path=["diagnoses", "patients", "prescriptions"],
                similarity_scores=similarity_scores_non_symmetric,
            ).to_dense()
        )

        self.assertTrue(
            torch.equal(
                computed_matrix_non_symmetric, self.expected_matrix_non_symmetric
            )
        )

    def test_compute_all_subgraphs(self):
        computed_subgraphs_naive = self.hsgnn_preprocessor.compute_all_subgraphs(
            approach="naive"
        )
        computed_subgraphs_fast = self.hsgnn_preprocessor.compute_all_subgraphs(
            approach="fast"
        )

        self.assertEqual(len(computed_subgraphs_naive), 5)
        self.assertEqual(len(computed_subgraphs_fast), 5)
        for subgraph in computed_subgraphs_naive:
            self.assertIsInstance(subgraph, torch.Tensor)

        for subgraph in computed_subgraphs_fast:
            self.assertIsInstance(subgraph, torch.Tensor)

        with self.assertRaises(AssertionError):
            self.hsgnn_preprocessor.compute_all_subgraphs(approach="invalid_approach")


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestHSGNNPreprocessor)
    unittest.TextTestRunner(verbosity=2).run(run_test)

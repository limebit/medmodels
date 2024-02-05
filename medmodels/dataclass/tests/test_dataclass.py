import numpy as np
import pandas as pd
import unittest

from medmodels.dataclass.dataclass import MedRecord
from medmodels.dataclass.utils import df_to_nodes, df_to_edges


class TestDataClass(unittest.TestCase):
    def setUp(self):
        self.medrecord = MedRecord()
        patients = pd.DataFrame(
            {"patients_id": ["P-1", "P-2"], "age": [30, 40], "sex": ["male", "female"]}
        )
        patient_nodes = df_to_nodes(patients, "patients_id", ["age", "sex"])

        diagnoses = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "code": [1, 2],
                "patients_id": ["P-1", "P-2"],
            }
        )
        diagnoses_nodes = df_to_nodes(diagnoses, "diagnoses_id", ["code"])
        diagnoses_patient_edges = df_to_edges(
            diagnoses, "diagnoses_id", "patients_id", []
        )

        self.assertEqual(self.medrecord.groups, [])
        self.medrecord.add_nodes(patient_nodes, "patients")
        self.medrecord.add_nodes(diagnoses_nodes, "diagnoses")
        self.medrecord.add_edges(diagnoses_patient_edges)
        self.medrecord.add_group("test_group", identifier=["P-1", "D-1"])

    def test_nodes(self):
        self.assertEqual(
            self.medrecord.nodes, ["P-1", "P-2", "D-1", "D-2", "test_group"]
        )

    def test_node(self):
        self.assertEqual(
            self.medrecord.node("P-1"), {"P-1": {"age": 30, "sex": "male"}}
        )

    def test_edges(self):
        self.assertEqual(
            self.medrecord.edges,
            [
                "P-1 D-1 {}",
                "P-1 test_group {}",
                "P-2 D-2 {}",
                "D-1 test_group {}",
            ],
        )

    def test_edge(self):
        self.assertEqual(self.medrecord.edge("P-1", "D-1"), {0: {}})

    def test_groups(self):
        self.assertEqual(self.medrecord.groups, ["test_group"])

    def test_group(self):
        self.assertEqual(self.medrecord.group("test_group"), ["P-1", "D-1"])
        self.assertRaises(
            AssertionError,
            self.medrecord.group,
            "non_existing_group",
        )

    def test_group_add_remove(self):
        self.assertEqual(self.medrecord.group("test_group"), ["P-1", "D-1"])
        self.medrecord.remove_node_from_group("test_group", "D-1")
        self.assertEqual(self.medrecord.group("test_group"), ["P-1"])
        self.medrecord.add_group("test_group", ["D-1"])
        self.assertEqual(self.medrecord.group("test_group"), ["P-1", "D-1"])
        self.assertRaises(
            AssertionError,
            self.medrecord.remove_node_from_group,
            "any_group",
            "any_node",
        )
        self.assertRaises(
            AssertionError,
            self.medrecord.add_group,
            "any_group",
            identifier=["non_existing_node"],
        )
        self.assertRaises(
            AssertionError,
            self.medrecord.add_group,
            "any_group",
            identifier="node_as_str_instead_of_list",
        )
        # No identifier and no criteria
        self.assertRaises(
            AssertionError,
            self.medrecord.add_group,
            "any_group",
            identifier=[],
            criteria=[],
        )
        self.medrecord.add_group("another_test_group", criteria=["patients age > 35"])
        self.assertEqual(
            self.medrecord.group("another_test_group"),
            ["P-2"],
        )

        # Identifier and criteria
        self.medrecord.add_group(
            "last_test_group",
            identifier=["P-1", "D-1", "D-2"],
            criteria=["patients age > 35"],
        )
        # There is no node that satisfies both identifier and criteria (they have to
        # be patients and also have age > 35)
        self.assertEqual(
            self.medrecord.group("last_test_group"),
            [],
        )

        self.medrecord.add_group(
            "last_test_group", identifier=["P-1", "P-2"], criteria=["patients age > 35"]
        )
        self.assertEqual(
            self.medrecord.group("last_test_group"),
            ["P-2"],
        )

        # Identifier and criteria with multiple dimensions
        self.medrecord.add_group(
            "diagnose_group",
            identifier=["P-2", "D-1"],
            criteria=["diagnoses code == 1"],
        )
        self.assertEqual(
            self.medrecord.group("diagnose_group"),
            ["D-1"],
        )

        # Test remove group
        self.medrecord.remove_group("test_group")
        self.assertNotIn("test_group", self.medrecord.groups)
        self.assertNotIn("test_group", self.medrecord._node_mapping["__group__"])
        self.assertNotIn("test_group", self.medrecord.G.nodes)

        # We see that if we remove all groups, the __group__ dimension is also removed
        for group in self.medrecord.groups:
            self.medrecord.remove_group(group)
        self.assertEqual(self.medrecord.groups, [])
        self.assertNotIn("__group__", self.medrecord._node_mapping)

    def test_dimensions(self):
        self.assertEqual(self.medrecord.dimensions, ["patients", "diagnoses"])

    def test_dimension(self):
        self.assertEqual(self.medrecord.dimension("patients"), ["P-1", "P-2"])
        self.assertEqual(
            self.medrecord.dimension("patients", "diagnoses"),
            [
                "P-1",
                "P-2",
                "D-1",
                "D-2",
            ],
        )

    def test_get_dimension_name(self):
        self.assertEqual(self.medrecord.get_dimension_name("P-1"), "patients")
        self.assertRaises(
            KeyError,
            self.medrecord.get_dimension_name,
            "non_existing_node",
        )

    def test_neighbors(self):
        self.assertEqual(
            self.medrecord.neighbors("P-1").sort(), ["test_group", "D-1"].sort()
        )
        self.assertEqual(
            self.medrecord.neighbors("D-1", dimension_filter=["patients"]), ["P-1"]
        )
        self.assertEqual(
            self.medrecord.neighbors("D-1", "P-1").sort(),
            ["test_group", "D-1", "P-1"].sort(),
        )
        self.assertRaises(
            AssertionError, self.medrecord.neighbors, "P1", dimension_filter=["test"]
        )
        self.assertRaises(AssertionError, self.medrecord.neighbors, "non_existing_node")

    def test_add_remove_nodes(self):
        test_1 = pd.DataFrame({"test_1_id": ["N-1"], "age": [30], "sex": ["male"]})
        test_2 = pd.DataFrame({"test_2_id": ["M-1"], "age": [30], "sex": ["male"]})
        test_1_nodes = df_to_nodes(test_1, "test_1_id", ["age", "sex"])
        test_2_nodes = df_to_nodes(test_2, "test_2_id", ["age", "sex"])
        self.medrecord.add_nodes(test_1_nodes, "test_dimension")
        self.assertEqual(
            self.medrecord.node("N-1"), {"N-1": {"age": 30, "sex": "male"}}
        )
        # test if identifer of nodes must be unique
        self.assertRaises(
            AssertionError, self.medrecord.add_nodes, test_1_nodes, "test_1_nodes"
        )
        # test if appending to existing dimension will throw info logging
        with self.assertLogs() as captured:
            self.medrecord.add_nodes(test_2_nodes, "patients")
        print(captured.records[0].getMessage())
        self.assertEqual(
            captured.records[0].getMessage(),
            "Info: Dimension patients in use, will append data.",
        )

    def test_add_remove_edges(self):
        test_df = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "code": [1, 2],
                "patients_id": ["P-2", "P-1"],
            }
        )
        test_edges = df_to_edges(test_df, "diagnoses_id", "patients_id", [])
        self.assertNotIn("P-2 D-1 {}", self.medrecord.edges)
        self.medrecord.add_edges(test_edges)
        self.assertIn("P-2 D-1 {}", self.medrecord.edges)

    def test_utils(self):
        self.assertTrue(
            self.medrecord.is_unique(
                np.array([["non_added_node1"], ["non_added_node2"]])
            )
        )
        self.assertFalse(
            self.medrecord.is_unique(np.array([["same_node"], ["same_node"]]))
        )
        self.assertFalse(self.medrecord.is_unique(np.array([["P-1"]])))

    def test_edges_to_df(self):
        expected_edge_frame = pd.DataFrame(
            {
                "id1": ["P-1", "P-1", "P-2", "D-1"],
                "id2": ["D-1", "test_group", "D-2", "test_group"],
            }
        )
        pd.testing.assert_frame_equal(self.medrecord.edges_to_df(), expected_edge_frame)

        # Adding relation_type to the edges
        test_adding = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "code": [1, 2],
                "patients_id": ["P-2", "P-1"],
            }
        )

        test_adding["relation_type"] = "patients_diagnoses"
        diagnoses_patient_edges = df_to_edges(
            test_adding, "diagnoses_id", "patients_id", ["relation_type"]
        )
        self.medrecord.add_edges(diagnoses_patient_edges)
        expected_edge_frame = pd.DataFrame(
            {
                "id1": ["P-1", "P-1", pd.NA, "P-2", pd.NA, "D-1"],
                "id2": ["D-1", "test_group", pd.NA, "D-2", pd.NA, "test_group"],
                "patients_id": [pd.NA, pd.NA, "P-1", pd.NA, "P-2", pd.NA],
                "diagnoses_id": [pd.NA, pd.NA, "D-2", pd.NA, "D-1", pd.NA],
                "relation_type": [
                    pd.NA,
                    pd.NA,
                    "patients_diagnoses",
                    pd.NA,
                    "patients_diagnoses",
                    pd.NA,
                ],
            }
        )
        # Check exact is False because of the different NaN formats
        pd.testing.assert_frame_equal(
            self.medrecord.edges_to_df().sort_index(axis=1),
            expected_edge_frame.sort_index(axis=1),
            check_exact=False,
        )

    def test_nodes_to_df(self):
        expected_edge_frame = pd.DataFrame(
            {
                "patients_id": {0: "P-1", 1: "P-2"},
                "age": {0: 30, 1: 40},
                "sex": {0: "male", 1: "female"},
            }
        )
        pd.testing.assert_frame_equal(
            self.medrecord.nodes_to_df("patients"), expected_edge_frame
        )

    def test_dimensions_to_dict(self):
        # AssertionError raised if no relation_type is provided
        self.assertRaises(
            AssertionError,
            self.medrecord.dimensions_to_dict,
        )

        diagnoses_extra = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "patients_id": ["P-2", "P-1"],
                "relation_type": ["patients_diagnoses", "patients_diagnoses"],
            }
        )
        diagnoses_patient_edges = df_to_edges(
            diagnoses_extra, "diagnoses_id", "patients_id", ["relation_type"]
        )
        self.medrecord.add_edges(diagnoses_patient_edges)

        expected_patients = pd.DataFrame(
            {"patients_id": ["P-1", "P-2"], "age": [30, 40], "sex": ["male", "female"]}
        )
        expected_diagnoses = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "code": [1, 2],
                "patients_id": ["P-2", "P-1"],
            }
        )
        pd.testing.assert_frame_equal(
            self.medrecord.dimensions_to_dict()["patients"],
            expected_patients,
        )
        pd.testing.assert_frame_equal(
            self.medrecord.dimensions_to_dict()["diagnoses"],
            expected_diagnoses,
        )

    def test_to_df(self):
        diagnoses_extra = pd.DataFrame(
            {
                "diagnoses_id": ["D-1", "D-2"],
                "patients_id": ["P-2", "P-1"],
                "relation_type": ["patients_diagnoses", "patients_diagnoses"],
            }
        )
        diagnoses_patient_edges = df_to_edges(
            diagnoses_extra, "diagnoses_id", "patients_id", ["relation_type"]
        )
        self.medrecord.add_edges(diagnoses_patient_edges)

        expected_edge_frame = pd.DataFrame(
            {
                "type": ["patients", "patients", "diagnoses", "diagnoses"],
                "patients_id": ["P-1", "P-2", "P-2", "P-1"],
                "age": [30, 40, np.nan, np.nan],
                "sex": ["male", "female", np.nan, np.nan],
                "diagnoses_id": [np.nan, np.nan, "D-1", "D-2"],
                "code": [np.nan, np.nan, 1, 2],
            }
        )
        expected_edge_frame["age"] = expected_edge_frame["age"].astype(float)
        expected_edge_frame["code"] = expected_edge_frame["code"].astype(float)

        pd.testing.assert_frame_equal(
            self.medrecord.to_df().sort_index(axis=1),
            expected_edge_frame.sort_index(axis=1),
            check_exact=False,
        )

    def test_repr(self):
        string = (
            "MedicalRecord dimensions: patients (2 Nodes), "
            "diagnoses (2 Nodes), __group__ (1 Nodes)"
        )
        self.assertTrue(repr(self.medrecord) == string)

    def test_explorer(self):
        self.assertEqual(self.medrecord.explorer.patients.age, "patients age")
        self.assertEqual(self.medrecord.explorer.diagnoses.code, "diagnoses code")


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestDataClass)
    unittest.TextTestRunner(verbosity=2).run(run_test)

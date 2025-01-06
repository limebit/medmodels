import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

import medmodels as mm
from medmodels.medrecord._overview import extract_attribute_summary, prettify_table
from medmodels.medrecord.querying import EdgeOperand, NodeOperand


def create_medrecord():
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3"],
            "age": [20, 30, 70],
        }
    )

    diagnosis = pd.DataFrame({"index": ["D1", "D2"]})

    prescriptions = pd.DataFrame(
        {"index": ["M1", "M2", "M3"], "ATC": ["B01AF01", "B01AA03", np.nan]}
    )

    nodes = [patients, diagnosis, prescriptions]

    edges = pd.DataFrame(
        {
            "source": ["D1", "M1", "D1"],
            "target": ["P1", "P2", "P3"],
            "time": ["2000-01-01", "1999-10-15", "1999-12-15"],
        }
    )

    edges.time = pd.to_datetime(edges.time)

    edges_disease = pl.DataFrame(
        {
            "source": ["D1", "D1", "D1"],
            "target": ["P1", "P2", "P3"],
            "intensity": [1, "low", None],
        },
        strict=False,
    )

    groups = [
        ("Patients", patients["index"].to_list()),
        ("Stroke", ["D1"]),
        ("Medications", ["M1", "M2", "M3"]),
        ("Aspirin", ["M3"]),
    ]

    medrecord = mm.MedRecord.from_pandas(
        nodes=[(node, "index") for node in nodes],
        edges=(edges, "source", "target"),
    )

    medrecord.add_edges_polars(edges=(edges_disease, "source", "target"))

    edges_meds = pd.DataFrame(
        {
            "source": ["M1", "M2", "M3"],
            "target": ["P1", "P2", "P3"],
            "time": ["2000-01-01", "1999-10-15", "1999-12-15"],
        }
    )
    edges_meds["time"] = pd.to_datetime(edges_meds["time"])

    medrecord.add_edges_pandas(
        edges=(edges_meds, "source", "target"), group="patient-medications"
    )

    for group, group_list in groups:
        medrecord.add_group(group, group_list)

    return medrecord


class TestOverview(unittest.TestCase):
    def test_extract_attribute_summary(self):
        # medrecord without schema
        medrecord = create_medrecord()

        def query1(node: NodeOperand):
            node.in_group("Stroke")

        # No attributes
        no_attributes = extract_attribute_summary(medrecord.node[query1])

        self.assertDictEqual(no_attributes, {})

        def query2(node: NodeOperand):
            node.in_group("Patients")

        # numeric type
        numeric_attribute = extract_attribute_summary(medrecord.node[query2])

        numeric_expected = {
            "age": {"type": "Continuous", "min": 20, "max": 70, "mean": 40.0}
        }

        self.assertDictEqual(numeric_attribute, numeric_expected)

        def query3(node: NodeOperand):
            node.in_group("Medications")

        # string attributes
        str_attributes = extract_attribute_summary(medrecord.node[query3])

        self.assertDictEqual(
            str_attributes,
            {"ATC": {"type": "Categorical", "values": "Values: B01AA03, B01AF01"}},
        )

        def query4(node: NodeOperand):
            node.in_group("Aspirin")

        # nan attribute
        nan_attributes = extract_attribute_summary(medrecord.node[query4])

        self.assertDictEqual(nan_attributes, {"ATC": {"type": "-", "values": "-"}})

        def query5(edge: EdgeOperand):
            edge.source_node().in_group("Medications")
            edge.target_node().in_group("Patients")

        # temporal attributes
        temp_attributes = extract_attribute_summary(medrecord.edge[query5])

        self.assertDictEqual(
            temp_attributes,
            {
                "time": {
                    "type": "Temporal",
                    "max": datetime(2000, 1, 1, 0, 0),
                    "min": datetime(1999, 10, 15, 0, 0),
                }
            },
        )

        def query6(edge: EdgeOperand):
            edge.source_node().in_group("Stroke")
            edge.target_node().in_group("Patients")

        # mixed attributes
        mixed_attributes = extract_attribute_summary(
            medrecord.edge[medrecord.select_edges(query6)]
        )
        self.assertDictEqual(
            mixed_attributes,
            {
                "time": {
                    "type": "Temporal",
                    "min": datetime(1999, 12, 15, 0, 0),
                    "max": datetime(2000, 1, 1, 0, 0),
                },
                "intensity": {"type": "Categorical", "values": "Values: 1, low"},
            },
        )

        # with schema
        mr_schema = mm.MedRecord.from_example_dataset()
        nodes_schema = mr_schema.group("patient")["nodes"]

        node_info = extract_attribute_summary(
            mr_schema.node[nodes_schema],
            schema=mr_schema.schema.group("patient").nodes,
        )

        self.assertDictEqual(
            node_info,
            {
                "age": {"type": "Continuous", "min": 19, "max": 96, "mean": 43.20},
                "gender": {"type": "Categorical", "values": "Categories: F, M"},
            },
        )

        def query7(edge: EdgeOperand):
            edge.in_group("patient_diagnosis")

        # compare schema and not schema
        patient_diagnosis = extract_attribute_summary(
            mr_schema.edge[query7],
            schema=mr_schema.schema.group("patient_diagnosis").edges,
        )

        self.assertDictEqual(
            patient_diagnosis,
            {
                "diagnosis_time": {
                    "type": "Temporal",
                    "min": datetime(1962, 10, 21, 0, 0),
                    "max": datetime(2024, 4, 12, 0, 0),
                },
                "duration_days": {
                    "type": "Continuous",
                    "min": 0.0,
                    "max": 3416.0,
                    "mean": 405.0232558139535,
                },
            },
        )

    def test_prettify_table(self):
        medrecord = create_medrecord()

        header = ["group nodes", "count", "attribute", "type", "info"]

        expected_nodes = [
            "---------------------------------------------------------------------",
            "Group Nodes     Count Attribute Type        Info                     ",
            "---------------------------------------------------------------------",
            "Aspirin         1     ATC       -           -                        ",
            "Medications     3     ATC       Categorical Values: B01AA03, B01AF01 ",
            "Patients        3     age       Continuous  min: 20                  ",
            "                                            max: 70                  ",
            "                                            mean: 40.00              ",
            "Stroke          1     -         -           -                        ",
            "Ungrouped Nodes 1     -         -           -                        ",
            "---------------------------------------------------------------------",
        ]

        self.assertEqual(
            prettify_table(medrecord._describe_group_nodes(), header, decimal=2),
            expected_nodes,
        )

        header = ["group edges", "count", "attribute", "type", "info"]

        expected_edges = [
            "----------------------------------------------------------------------",
            "Group Edges         Count Attribute Type     Info                     ",
            "----------------------------------------------------------------------",
            "patient-medications 3     time      Temporal min: 1999-10-15 00:00:00 ",
            "                                             max: 2000-01-01 00:00:00 ",
            "Ungrouped Edges     6     -         -        -                        ",
            "----------------------------------------------------------------------",
        ]

        self.assertEqual(
            prettify_table(medrecord._describe_group_edges(), header, decimal=2),
            expected_edges,
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestOverview)
    unittest.TextTestRunner(verbosity=2).run(run_test)

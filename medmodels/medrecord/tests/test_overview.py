import unittest

import numpy as np
import pandas as pd
import polars as pl

import medmodels as mm
from medmodels.medrecord._overview import extract_attribute_summary, prettify_table
from medmodels.medrecord.querying import edge, node


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

    for group, group_list in groups:
        medrecord.add_group(group, group_list)

    return medrecord


class TestOverview(unittest.TestCase):
    def test_extract_attribute_summary(self):
        # medrecord without schema
        medrecord = create_medrecord()

        # No attributes
        no_attributes = extract_attribute_summary(
            medrecord.node[node().in_group("Stroke")]
        )

        self.assertDictEqual(no_attributes, {})

        # numeric type
        numeric_attribute = extract_attribute_summary(
            medrecord.node[node().in_group("Patients")]
        )

        numeric_expected = {"age": ["min: 20", "max: 70", "mean: 40.00"]}

        self.assertDictEqual(numeric_attribute, numeric_expected)

        # string attributes
        str_attributes = extract_attribute_summary(
            medrecord.node[node().in_group("Medications")]
        )

        self.assertDictEqual(str_attributes, {"ATC": ["Values: B01AA03, B01AF01"]})

        # nan attribute
        nan_attributes = extract_attribute_summary(
            medrecord.node[node().in_group("Aspirin")]
        )
        self.assertDictEqual(nan_attributes, {"ATC": ["-"]})

        # temporal attributes
        temp_attributes = extract_attribute_summary(
            medrecord.edge[
                medrecord.select_edges(
                    edge().connected_source_with(node().in_group("Medications"))
                    & edge().connected_target_with(node().in_group("Patients"))
                )
            ]
        )

        self.assertDictEqual(
            temp_attributes,
            {"time": ["min: 1999-10-15 00:00:00", "max: 1999-10-15 00:00:00"]},
        )

        # mixed attributes
        mixed_attributes = extract_attribute_summary(
            medrecord.edge[
                medrecord.select_edges(
                    edge().connected_source_with(node().in_group("Stroke"))
                    & edge().connected_target_with(node().in_group("Patients"))
                )
            ]
        )
        self.assertDictEqual(
            mixed_attributes,
            {
                "time": ["min: 1999-12-15 00:00:00", "max: 2000-01-01 00:00:00"],
                "intensity": ["Values: 1, low"],
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
                "age": ["min: 19", "max: 96", "mean: 43.20"],
                "gender": ["Categories: F, M"],
            },
        )

        # compare schema and not schema
        patient_diagnosis = extract_attribute_summary(
            mr_schema.edge[edge().in_group("patient_diagnosis")],
            schema=mr_schema.schema.group("patient_diagnosis").edges,
        )

        self.assertDictEqual(
            patient_diagnosis,
            {
                "diagnosis_time": [
                    "min: 1962-10-21 00:00:00",
                    "max: 2024-04-12 00:00:00",
                ],
                "duration_days": [
                    "min: 0.0",
                    "max: 3416.0",
                    "mean: 405.02",
                ],
            },
        )

    def test_prettify_table(self):
        medrecord = create_medrecord()

        header = ["group nodes", "count", "attribute", "info"]

        expected_empty = [
            "---------------------------------------------------------",
            "Group Nodes     Count Attribute Info                     ",
            "---------------------------------------------------------",
            "Aspirin         1     ATC       -                        ",
            "Medications     3     ATC       Values: B01AA03, B01AF01 ",
            "Patients        3     age       min: 20                  ",
            "                                max: 70                  ",
            "                                mean: 40.00              ",
            "Stroke          1     -         -                        ",
            "Ungrouped Nodes 1     -         -                        ",
            "---------------------------------------------------------",
        ]

        self.assertEqual(
            prettify_table(medrecord._describe_group_nodes(), header), expected_empty
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestOverview)
    unittest.TextTestRunner(verbosity=2).run(run_test)

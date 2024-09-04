import unittest

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

        self.assertTrue(
            no_attributes.equals(pl.DataFrame({"Attribute": "-", "Info": "-"}))
        )

        def query2(node: NodeOperand):
            node.in_group("Patients")

        # numeric type
        numeric_attribute = extract_attribute_summary(medrecord.node[query2])

        numeric_expected = pl.DataFrame(
            {"Attribute": ["age"] * 3, "Info": ["min: 20", "max: 70", "mean: 40.00"]}
        )

        self.assertTrue(numeric_attribute.equals(numeric_expected))

        def query3(node: NodeOperand):
            node.in_group("Medications")

        # string attributes
        str_attributes = extract_attribute_summary(medrecord.node[query3])

        self.assertTrue(
            str_attributes.equals(
                pl.DataFrame({"Attribute": "ATC", "Info": "Values: B01AA03, B01AF01"})
            )
        )

        def query4(node: NodeOperand):
            node.in_group("Aspirin")

        # nan attribute
        nan_attributes = extract_attribute_summary(medrecord.node[query4])

        self.assertTrue(
            nan_attributes.equals(pl.DataFrame({"Attribute": "ATC", "Info": "-"}))
        )

        def query5(edge: EdgeOperand):
            edge.source_node().in_group("Medications")
            edge.target_node().in_group("Patients")

        # temporal attributes
        temp_attributes = extract_attribute_summary(medrecord.edge[query5])

        self.assertTrue(
            temp_attributes.equals(
                pl.DataFrame(
                    {
                        "Attribute": ["time"] * 2,
                        "Info": [
                            "min: 1999-10-15 00:00:00",
                            "max: 1999-10-15 00:00:00",
                        ],
                    }
                )
            )
        )

        def query6(edge: EdgeOperand):
            edge.source_node().in_group("Stroke")
            edge.target_node().in_group("Patients")

        # mixed attributes
        mixed_attributes = extract_attribute_summary(
            medrecord.edge[medrecord.select_edges(query6)]
        )

        expected_mixed = pl.DataFrame(
            {
                "Attribute": ["intensity", "time", "time"],
                "Info": [
                    "Values: 1, low",
                    "min: 1999-12-15 00:00:00",
                    "max: 2000-01-01 00:00:00",
                ],
            }
        )

        self.assertTrue(mixed_attributes.equals(expected_mixed))

        # with schema
        mr_schema = mm.MedRecord.from_example_dataset()
        nodes_schema = mr_schema.group("patient")["nodes"]

        node_info = extract_attribute_summary(
            mr_schema.node[nodes_schema],
            schema=mr_schema.schema.group("patient").nodes,
        )

        expected_info = pl.DataFrame(
            {
                "Attribute": ["age", "age", "age", "gender"],
                "Info": ["min: 19", "max: 96", "mean: 43.20", "Categories: F, M"],
            }
        )

        self.assertTrue(node_info.equals(expected_info))

        def query7(edge: EdgeOperand):
            edge.in_group("patient_diagnosis")

        # compare schema and not schema
        patient_diagnosis = extract_attribute_summary(
            mr_schema.edge[query7],
            schema=mr_schema.schema.group("patient_diagnosis").edges,
        )

        patient_diagnoses_expected = pl.DataFrame(
            {
                "Attribute": [
                    "diagnosis_time",
                    "diagnosis_time",
                    "duration_days",
                    "duration_days",
                    "duration_days",
                ],
                "Info": [
                    "min: 1962-10-21 00:00:00",
                    "max: 2024-04-12 00:00:00",
                    "min: 0.0",
                    "max: 3416.0",
                    "mean: 405.02",
                ],
            }
        )

        self.assertTrue(patient_diagnosis.equals(patient_diagnoses_expected))

    def test_prettify_table(self):
        df_empty = pl.DataFrame(
            {
                "Attribute": ["-"],
                "Info": ["-"],
            }
        )

        expected_list_empty = [
            "----------------",
            "Attribute Info",
            "----------------",
            "-         -    ",
            "----------------",
        ]

        self.assertEqual(prettify_table(df_empty), expected_list_empty)

        df = pl.DataFrame(
            {
                "Attribute": ["age", "age", "age", "gender"],
                "Info": ["min: 19", "max: 96", "mean: 43.20", "Categories: F, M"],
            }
        )

        expected_list = [
            "----------------------------",
            "Attribute Info            ",
            "----------------------------",
            "age       min: 19          ",
            "          max: 96          ",
            "          mean: 43.20      ",
            "gender    Categories: F, M ",
            "----------------------------",
        ]

        self.assertEqual(prettify_table(df), expected_list)


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestOverview)
    unittest.TextTestRunner(verbosity=2).run(run_test)

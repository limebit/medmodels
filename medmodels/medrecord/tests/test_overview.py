import unittest

import pandas as pd
import polars as pl

import medmodels as mm
from medmodels.medrecord.overview import extract_attribute_summary
from medmodels.medrecord.querying import edge, node


def create_medrecord():
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            "age": [20, 30, 70, 30, 40, 50, 60, 70, 80],
        }
    )
    diagnosis = pd.DataFrame({"index": ["D1", "D2"]})
    prescriptions = pd.DataFrame({"index": ["M1", "M2"], "ATC": ["B01AF01", "B01AA03"]})
    nodes = [patients, diagnosis, prescriptions]
    edges = pd.DataFrame(
        {
            "source": [
                "D1",
                "M2",
                "M1",
                "M2",
                "D1",
                "D1",
                "D1",
                "M1",
                "D1",
                "M2",
                "M1",
                "D1",
                "M2",
            ],
            "target": [
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P2",
                "P3",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P9",
            ],
            "time": [
                "2000-01-01",
                "1999-10-15",
                "2000-01-01",
                "1999-12-15",
                "1999-12-15",
                "2000-07-01",
                "2000-01-05",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
            ],
        }
    )
    groups = [
        ("Patients", patients["index"].to_list()),
        ("Stroke", ["D1"]),
        ("Headache", ["D2"]),
        ("Rivaroxaban", ["M1"]),
        ("Warfarin", ["M2"]),
        ("Medications", ["M1", "M2"]),
    ]
    medrecord = mm.MedRecord.from_pandas(
        nodes=[(node, "index") for node in nodes],
        edges=(edges, "source", "target"),
    )
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
        self.assertTrue(
            no_attributes.equals(pl.DataFrame({"Attribute": "-", "Info": "-"}))
        )

        # numeric type
        numeric_attribute = extract_attribute_summary(
            medrecord.node[node().in_group("Patients")]
        )
        numeric_expected = pl.DataFrame(
            {"Attribute": ["age"] * 3, "Info": ["min: 20", "max: 80", "mean: 50.00"]}
        )
        self.assertTrue(numeric_attribute.equals(numeric_expected))

        # temporal
        edges = medrecord.select_edges(
            edge().connected_source_with(node().in_group("Rivaroxaban"))
            & edge().connected_target_with(node().in_group("Patients"))
        )
        temporal_attributes = extract_attribute_summary(medrecord.edge[edges])
        temporal_expected = pl.DataFrame(
            {
                "Attribute": ["time"] * 2,
                "Info": [
                    "min: 2000-01-01 00:00:00",
                    "max: 2000-01-01 00:00:00",
                ],
            }
        )
        self.assertTrue(temporal_attributes.equals(temporal_expected))
        # string attributes
        str_attributes = extract_attribute_summary(
            medrecord.node[node().in_group("Medications")]
        )
        self.assertTrue(
            str_attributes.equals(
                pl.DataFrame({"Attribute": "ATC", "Info": "Values: B01AA03, B01AF01"})
            )
        )

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

        # compare schema and not schema
        schema_attributes = extract_attribute_summary(
            mr_schema.edge[edge().in_group("patient_diagnosis")]
        )
        no_schema_attributes = extract_attribute_summary(
            mr_schema.edge[
                mr_schema.select_edges(
                    edge().connected_source_with(node().in_group("patient"))
                    & edge().connected_target_with(node().in_group("diagnosis"))
                )
            ]
        )
        self.assertTrue(schema_attributes.equals(no_schema_attributes))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestOverview)
    unittest.TextTestRunner(verbosity=2).run(run_test)

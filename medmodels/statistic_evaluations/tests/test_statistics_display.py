import unittest

import numpy as np
import pandas as pd
import polars as pl

import medmodels as mm
from medmodels.statistic_evaluations.statistical_analysis.statistics_display import (
    prettify_table,
)


def create_medrecord() -> mm.MedRecord:
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3"],
            "age": [20, 30, 70],
        }
    )

    diagnosis = pd.DataFrame({"index": ["D1", "D2"]})

    prescriptions = pd.DataFrame(
        {
            "index": ["M1", "M2", "M3"],
            "ATC": ["B01AF01", "B01AA03", np.nan],
        }
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


class TestStatisticsDisplay(unittest.TestCase):
    def test_prettify_table(self) -> None:
        medrecord = create_medrecord()

        header = ["group nodes", "count", "attribute", "type", "data"]

        expected_nodes = [
            "---------------------------------------------------------------------",
            "Group Nodes     Count Attribute Type        Data                     ",
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

        assert (
            prettify_table(medrecord._describe_group_nodes(), header, decimal=2)
            == expected_nodes
        )

        header = ["group edges", "count", "attribute", "type", "info"]

        expected_edges = [
            "-----------------------------------------------------------------------",
            "Group Edges         Count Attribute Type     Info                      ",
            "-----------------------------------------------------------------------",
            "patient-medications 3     time      Temporal min: 1999-10-15 00:00:00  ",
            "                                             max: 2000-01-01 00:00:00  ",
            "                                             mean: 1999-11-30 08:00:00 ",
            "Ungrouped Edges     6     -         -        -                         ",
            "-----------------------------------------------------------------------",
        ]

        assert (
            prettify_table(medrecord._describe_group_edges(), header, decimal=2)
            == expected_edges
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestStatisticsDisplay)
    unittest.TextTestRunner(verbosity=2).run(run_test)

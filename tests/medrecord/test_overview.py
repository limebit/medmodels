import unittest
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table

import medmodels as mm
from medmodels import MedRecord
from medmodels.medrecord._overview import (
    Metric,
    TypeTable,
    get_attribute_metric,
    get_values_from_attribute,
    join_tables_with_titles,
    prettify_table,
)
from medmodels.medrecord.querying import (
    EdgeOperand,
    NodeOperand,
)
from tests.medrecord.test_medrecord import strip_ansi


def repr_table(table: Table) -> str:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    console.print(table)

    return strip_ansi(buffer.getvalue())


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


class TestOverview(unittest.TestCase):
    def test_prettify_table(self) -> None:
        header = ["Group Nodes", "count", "attribute", "type", "datatype", "data"]

        table = prettify_table({}, header, decimal=2, type_table=TypeTable.MedRecord)

        expected = "\n".join(
            [
                "┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━┓",
                "┃ Group Nodes ┃ count ┃ attribute ┃ type ┃ datatype ┃ data ┃",
                "┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━┩",
                "│ No data     │       │           │      │          │      │",
                "└─────────────┴───────┴───────────┴──────┴──────────┴──────┘",
            ]
        )

        assert repr_table(table) == expected

        medrecord = MedRecord()
        medrecord.add_nodes([(0, {})])
        medrecord.add_group("group1", nodes=[0])

        table = prettify_table(
            medrecord._describe_group_nodes(),
            header,
            decimal=2,
            type_table=TypeTable.MedRecord,
        )

        expected = "\n".join(
            [
                "┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━┓",
                "┃ Group Nodes ┃ count ┃ attribute     ┃ type ┃ datatype ┃ data ┃",
                "┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━┩",
                "│ group1      │ 1     │ No attributes │      │          │      │",
                "└─────────────┴───────┴───────────────┴──────┴──────────┴──────┘",
            ]
        )

        assert repr_table(table) == expected

        medrecord = MedRecord.from_simple_example_dataset()

        expected = "\n".join(
            [
                "┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓",
                "┃ Group Nodes ┃ count ┃ attribute   ┃ type         ┃ datatype ┃ data             ┃",
                "┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩",
                "│ diagnosis   │ 25    │ description │ Unstructured │ String   │ -                │",
                "│             │       │             │              │          │                  │",
                "│ drug        │ 19    │ description │ Unstructured │ String   │ -                │",
                "│             │       │             │              │          │                  │",
                "│ patient     │ 5     │ age         │ Continuous   │ Int      │ min: 19          │",
                "│             │       │             │              │          │ mean: 43.2       │",
                "│             │       │             │              │          │ max: 96          │",
                "│             │       │ gender      │ Categorical  │ String   │ Categories: F, M │",
                "│             │       │             │              │          │                  │",
                "│ procedure   │ 24    │ description │ Unstructured │ String   │ -                │",
                "└─────────────┴───────┴─────────────┴──────────────┴──────────┴──────────────────┘",
            ]
        )
        table = prettify_table(
            medrecord._describe_group_nodes(),
            header,
            decimal=2,
            type_table=TypeTable.MedRecord,
        )

        assert repr_table(table) == expected

        header = ["Group Edges", "count", "attribute", "type", "datatype", "data"]

        expected = "\n".join(
            [
                "┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
                "┃ Group Edges       ┃ count ┃ attribute        ┃ type       ┃ datatype      ┃ data                      ┃",
                "┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩",
                "│ patient_diagnosis │ 60    │ duration_days    │ Continuous │ Option(Float) │ min: 0.0                  │",
                "│                   │       │                  │            │               │ mean: 405.02              │",
                "│                   │       │                  │            │               │ max: 3416.0               │",
                "│                   │       │ time             │ Temporal   │ DateTime      │ min: 1962-10-21 00:00:00  │",
                "│                   │       │                  │            │               │ mean: 2012-01-25 11:12:00 │",
                "│                   │       │                  │            │               │ max: 2024-04-12 00:00:00  │",
                "│                   │       │                  │            │               │                           │",
                "│ patient_drug      │ 50    │ cost             │ Continuous │ Float         │ min: 0.1                  │",
                "│                   │       │                  │            │               │ mean: 412.1               │",
                "│                   │       │                  │            │               │ max: 7822.2               │",
                "│                   │       │ quantity         │ Continuous │ Int           │ min: 1                    │",
                "│                   │       │                  │            │               │ mean: 2.96                │",
                "│                   │       │                  │            │               │ max: 12                   │",
                "│                   │       │ time             │ Temporal   │ DateTime      │ min: 1995-03-26 02:00:40  │",
                "│                   │       │                  │            │               │ mean: 2016-02-17 14:41:36 │",
                "│                   │       │                  │            │               │ max: 2024-04-12 11:59:55  │",
                "│                   │       │                  │            │               │                           │",
                "│ patient_procedure │ 50    │ duration_minutes │ Continuous │ Float         │ min: 4.0                  │",
                "│                   │       │                  │            │               │ mean: 19.44               │",
                "│                   │       │                  │            │               │ max: 59.0                 │",
                "│                   │       │ time             │ Temporal   │ DateTime      │ min: 1993-03-14 02:42:31  │",
                "│                   │       │                  │            │               │ mean: 2015-07-29 12:05:08 │",
                "│                   │       │                  │            │               │ max: 2024-04-24 03:38:35  │",
                "└───────────────────┴───────┴──────────────────┴────────────┴───────────────┴───────────────────────────┘",
            ]
        )

        table = prettify_table(
            medrecord._describe_group_edges(),
            header,
            decimal=2,
            type_table=TypeTable.MedRecord,
        )

        assert repr_table(table) == expected

        header = ["Group Edges", "attribute", "type", "datatype"]

        expected = "\n".join(
            [
                "┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┓",
                "┃ Group Edges ┃ attribute   ┃ type         ┃ datatype ┃",
                "┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━┩",
                "│ diagnosis   │ description │ Unstructured │ String   │",
                "│             │             │              │          │",
                "│ drug        │ description │ Unstructured │ String   │",
                "│             │             │              │          │",
                "│ patient     │ age         │ Continuous   │ Int      │",
                "│             │ gender      │ Categorical  │ String   │",
                "│             │             │              │          │",
                "│ procedure   │ description │ Unstructured │ String   │",
                "└─────────────┴─────────────┴──────────────┴──────────┘",
            ]
        )

        table = prettify_table(
            medrecord.get_schema()._describe_nodes_schema(),
            header,
            decimal=1,
            type_table=TypeTable.Schema,
        )

        assert repr_table(table) == expected

    def test_join_tables_with_titles(self) -> None:
        """Tests joining two simple tables with titles."""
        table1 = Table(show_lines=True)
        table1.add_column("ID", style="dim", width=6)
        table1.add_column("Value")
        table1.add_row("A", "Apple")

        table2 = Table(show_lines=True)
        table2.add_column("Key")
        table2.add_column("Data")
        table2.add_column("Flag")
        table2.add_row("X1", "TestData", "True")
        table2.add_row("Y2", "MoreData", "False")

        title1 = "First Section"
        title2 = "Second Section"

        joined_output = join_tables_with_titles(title1, table1, title2, table2)
        joined_output = strip_ansi(joined_output)

        expected = "\n".join(
            [
                "──────────────────────────────── First Section ─────────────────────────────────",
                "┏━━━━━━━━┳━━━━━━━┓",
                "┃ ID     ┃ Value ┃",
                "┡━━━━━━━━╇━━━━━━━┩",
                "│ A      │ Apple │",
                "└────────┴───────┘",
                "──────────────────────────────── Second Section ────────────────────────────────",
                "┏━━━━━┳━━━━━━━━━━┳━━━━━━━┓",
                "┃ Key ┃ Data     ┃ Flag  ┃",
                "┡━━━━━╇━━━━━━━━━━╇━━━━━━━┩",
                "│ X1  │ TestData │ True  │",
                "├─────┼──────────┼───────┤",
                "│ Y2  │ MoreData │ False │",
                "└─────┴──────────┴───────┘",
            ]
        )

        assert joined_output.strip() == expected.strip()

    def test_get_attribute_metric(self) -> None:
        medrecord = MedRecord.from_simple_example_dataset()

        def query_males(node: NodeOperand) -> None:
            node.attribute("gender").equal_to("M")

        result = get_attribute_metric(
            medrecord,
            query=query_males,
            attribute="age",
            metric=Metric.min,
            type="nodes",
        )
        assert result == 19

        result = get_attribute_metric(
            medrecord,
            query=query_males,
            attribute="age",
            metric=Metric.max,
            type="nodes",
        )
        assert result == 42

        result = get_attribute_metric(
            medrecord,
            query=query_males,
            attribute="age",
            metric=Metric.mean,
            type="nodes",
        )
        result = round(float(result), 2)  # pyright: ignore[reportArgumentType]

        assert result == 32.67

        def query_edges(edge: EdgeOperand) -> None:
            edge.in_group("patient_diagnosis")

        result = get_attribute_metric(
            medrecord,
            query=query_edges,
            attribute="duration_days",
            metric=Metric.min,
            type="edges",
        )
        assert result == 0

        result = get_attribute_metric(
            medrecord,
            query=query_edges,
            attribute="time",
            metric=Metric.max,
            type="edges",
        )
        assert result == datetime(2024, 4, 12, 0, 0)

        result = get_attribute_metric(
            medrecord,
            query=query_edges,
            attribute="duration_days",
            metric=Metric.mean,
            type="edges",
        )
        result = round(float(result), 2)  # pyright: ignore[reportArgumentType]

        assert result == 405.02

    def test_get_values_from_attribute(self) -> None:
        # Test with a MedRecord object
        medrecord = create_medrecord()

        def query_edges(edge: EdgeOperand) -> None:
            edge.in_group("patient-medications")

        result = get_values_from_attribute(
            medrecord, query=query_edges, attribute="time", type="edges"
        )
        assert result == {
            datetime(2000, 1, 1, 0, 0),
            datetime(1999, 10, 15, 0, 0),
            datetime(1999, 12, 15, 0, 0),
        }

        def query_nodes(node: NodeOperand) -> None:
            node.in_group("Patients")

        result = get_values_from_attribute(
            medrecord, query=query_nodes, attribute="age", type="nodes"
        )
        assert result == {20, 30, 70}


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestOverview)
    unittest.TextTestRunner(verbosity=2).run(run_test)

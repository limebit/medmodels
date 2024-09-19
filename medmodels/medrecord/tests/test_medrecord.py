import tempfile
import unittest
from typing import List, Tuple

import pandas as pd
import polars as pl

import medmodels.medrecord as mr
from medmodels import MedRecord
from medmodels.medrecord import edge as edge_select
from medmodels.medrecord import node as node_select
from medmodels.medrecord.types import Attributes, NodeIndex


def create_nodes() -> List[Tuple[NodeIndex, Attributes]]:
    return [
        ("0", {"lorem": "ipsum", "dolor": "sit"}),
        ("1", {"amet": "consectetur"}),
        ("2", {"adipiscing": "elit"}),
        ("3", {}),
    ]


def create_edges() -> List[Tuple[NodeIndex, NodeIndex, Attributes]]:
    return [
        ("0", "1", {"sed": "do", "eiusmod": "tempor"}),
        ("1", "0", {"sed": "do", "eiusmod": "tempor"}),
        ("1", "2", {"incididunt": "ut"}),
        ("0", "3", {}),
    ]


def create_pandas_nodes_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": ["0", "1"],
            "attribute": [1, 2],
        }
    )


def create_second_pandas_nodes_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": ["2", "3"],
            "attribute": [2, 3],
        }
    )


def create_pandas_edges_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["0", "1"],
            "target": ["1", "0"],
            "attribute": [1, 2],
        }
    )


def create_second_pandas_edges_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["0", "1"],
            "target": ["1", "0"],
            "attribute": [2, 3],
        }
    )


def create_medrecord() -> MedRecord:
    return MedRecord.from_tuples(create_nodes(), create_edges())


class TestMedRecord(unittest.TestCase):
    def test_from_tuples(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())

    def test_invalid_from_tuples(self):
        nodes = create_nodes()

        # Adding an edge pointing to a non-existent node should fail
        with self.assertRaises(IndexError):
            MedRecord.from_tuples(nodes, [("0", "50", {})])

        # Adding an edge from a non-existing node should fail
        with self.assertRaises(IndexError):
            MedRecord.from_tuples(nodes, [("50", "0", {})])

    def test_from_pandas(self):
        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
        )

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(0, medrecord.edge_count())

        medrecord = MedRecord.from_pandas(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
        )

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(0, medrecord.edge_count())

        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
            (create_pandas_edges_dataframe(), "source", "target"),
        )

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

        medrecord = MedRecord.from_pandas(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
            (create_pandas_edges_dataframe(), "source", "target"),
        )

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ],
        )

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())

        medrecord = MedRecord.from_pandas(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ],
        )

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())

    def test_from_polars(self):
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())
        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        medrecord = MedRecord.from_polars((nodes, "index"), (edges, "source", "target"))

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

        medrecord = MedRecord.from_polars(
            [(nodes, "index"), (second_nodes, "index")], (edges, "source", "target")
        )

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

        medrecord = MedRecord.from_polars(
            (nodes, "index"),
            [(edges, "source", "target"), (second_edges, "source", "target")],
        )

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())

        medrecord = MedRecord.from_polars(
            [(nodes, "index"), (second_nodes, "index")],
            [(edges, "source", "target"), (second_edges, "source", "target")],
        )

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())

    def test_invalid_from_polars(self):
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())
        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        # Providing the wrong node index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars((nodes, "invalid"), (edges, "source", "target"))

        # Providing the wrong node index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(
                [(nodes, "index"), (second_nodes, "invalid")],
                (edges, "source", "target"),
            )

        # Providing the wrong source index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars((nodes, "index"), (edges, "invalid", "target"))

        # Providing the wrong source index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(
                (nodes, "index"),
                [(edges, "source", "target"), (second_edges, "invalid", "target")],
            )

        # Providing the wrong target index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars((nodes, "index"), (edges, "source", "invalid"))

        # Providing the wrong target index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(
                (nodes, "index"),
                [(edges, "source", "target"), (edges, "source", "invalid")],
            )

    def test_from_example_dataset(self):
        medrecord = MedRecord.from_example_dataset()

        self.assertEqual(73, medrecord.node_count())
        self.assertEqual(160, medrecord.edge_count())

        self.assertEqual(25, len(medrecord.nodes_in_group("diagnosis")))
        self.assertEqual(19, len(medrecord.nodes_in_group("drug")))
        self.assertEqual(5, len(medrecord.nodes_in_group("patient")))
        self.assertEqual(24, len(medrecord.nodes_in_group("procedure")))

    def test_ron(self):
        medrecord = create_medrecord()

        with tempfile.NamedTemporaryFile() as f:
            medrecord.to_ron(f.name)

            loaded_medrecord = MedRecord.from_ron(f.name)

        self.assertEqual(medrecord.node_count(), loaded_medrecord.node_count())
        self.assertEqual(medrecord.edge_count(), loaded_medrecord.edge_count())

    def test_schema(self):
        schema = mr.Schema(
            groups={
                "group": mr.GroupSchema(
                    nodes={"attribute2": mr.Int()}, edges={"attribute2": mr.Int()}
                )
            },
            default=mr.GroupSchema(
                nodes={"attribute": mr.Int()}, edges={"attribute": mr.Int()}
            ),
        )

        medrecord = MedRecord.with_schema(schema)
        medrecord.add_group("group")

        medrecord.add_node("0", {"attribute": 1})

        with self.assertRaises(ValueError):
            medrecord.add_node("1", {"attribute": "1"})

        medrecord.add_node("1", {"attribute": 1, "attribute2": 1})

        medrecord.add_node_to_group("group", "1")

        medrecord.add_node("2", {"attribute": 1, "attribute2": "1"})

        with self.assertRaises(ValueError):
            medrecord.add_node_to_group("group", "2")

        medrecord.add_edge("0", "1", {"attribute": 1})

        with self.assertRaises(ValueError):
            medrecord.add_edge("0", "1", {"attribute": "1"})

        edge_index = medrecord.add_edge("0", "1", {"attribute": 1, "attribute2": 1})

        medrecord.add_edge_to_group("group", edge_index)

        edge_index = medrecord.add_edge("0", "1", {"attribute": 1, "attribute2": "1"})

        with self.assertRaises(ValueError):
            medrecord.add_edge_to_group("group", edge_index)

    def test_nodes(self):
        medrecord = create_medrecord()

        nodes = [x[0] for x in create_nodes()]

        for node in medrecord.nodes:
            self.assertTrue(node in nodes)

    def test_edges(self):
        medrecord = create_medrecord()

        edges = list(range(len(create_edges())))

        for edge in medrecord.edges:
            self.assertTrue(edge in edges)

    def test_groups(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual(["0"], medrecord.groups)

    def test_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual({"nodes": [], "edges": []}, medrecord.group("0"))

        medrecord.add_group("1", ["0"], [0])

        self.assertEqual({"nodes": ["0"], "edges": [0]}, medrecord.group("1"))

        self.assertEqual(
            {"0": {"nodes": [], "edges": []}, "1": {"nodes": ["0"], "edges": [0]}},
            medrecord.group(["0", "1"]),
        )

    def test_invalid_group(self):
        medrecord = create_medrecord()

        # Querying a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.group("0")

        medrecord.add_group("1", ["0"])

        # Querying a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.group(["0", "50"])

    def test_outgoing_edges(self):
        medrecord = create_medrecord()

        edges = medrecord.outgoing_edges("0")

        self.assertEqual(
            sorted([0, 3]),
            sorted(edges),
        )

        edges = medrecord.outgoing_edges(["0", "1"])

        self.assertEqual(
            {"0": sorted([0, 3]), "1": [1, 2]},
            {key: sorted(value) for (key, value) in edges.items()},
        )

        edges = medrecord.outgoing_edges(node_select().index().is_in(["0", "1"]))

        self.assertEqual(
            {"0": sorted([0, 3]), "1": [1, 2]},
            {key: sorted(value) for (key, value) in edges.items()},
        )

    def test_invalid_outgoing_edges(self):
        medrecord = create_medrecord()

        # Querying outgoing edges of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.outgoing_edges("50")

        # Querying outgoing edges of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.outgoing_edges(["0", "50"])

    def test_incoming_edges(self):
        medrecord = create_medrecord()

        edges = medrecord.incoming_edges("1")

        self.assertEqual([0], edges)

        edges = medrecord.incoming_edges(["1", "2"])

        self.assertEqual({"1": [0], "2": [2]}, edges)

        edges = medrecord.incoming_edges(node_select().index().is_in(["1", "2"]))

        self.assertEqual({"1": [0], "2": [2]}, edges)

    def test_invalid_incoming_edges(self):
        medrecord = create_medrecord()

        # Querying incoming edges of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.incoming_edges("50")

        # Querying incoming edges of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.incoming_edges(["0", "50"])

    def test_edge_endpoints(self):
        medrecord = create_medrecord()

        endpoints = medrecord.edge_endpoints(0)

        self.assertEqual(("0", "1"), endpoints)

        endpoints = medrecord.edge_endpoints([0, 1])

        self.assertEqual({0: ("0", "1"), 1: ("1", "0")}, endpoints)

        endpoints = medrecord.edge_endpoints(edge_select().index().is_in([0, 1]))

        self.assertEqual({0: ("0", "1"), 1: ("1", "0")}, endpoints)

    def test_invalid_edge_endpoints(self):
        medrecord = create_medrecord()

        # Querying endpoints of a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.edge_endpoints(50)

        # Querying endpoints of a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.edge_endpoints([0, 50])

    def test_edges_connecting(self):
        medrecord = create_medrecord()

        edges = medrecord.edges_connecting("0", "1")

        self.assertEqual([0], edges)

        edges = medrecord.edges_connecting(["0", "1"], "1")

        self.assertEqual([0], edges)

        edges = medrecord.edges_connecting(node_select().index().is_in(["0", "1"]), "1")

        self.assertEqual([0], edges)

        edges = medrecord.edges_connecting("0", ["1", "3"])

        self.assertEqual(sorted([0, 3]), sorted(edges))

        edges = medrecord.edges_connecting("0", node_select().index().is_in(["1", "3"]))

        self.assertEqual(sorted([0, 3]), sorted(edges))

        edges = medrecord.edges_connecting(["0", "1"], ["1", "2", "3"])

        self.assertEqual(sorted([0, 2, 3]), sorted(edges))

        edges = medrecord.edges_connecting(
            node_select().index().is_in(["0", "1"]),
            node_select().index().is_in(["1", "2", "3"]),
        )

        self.assertEqual(sorted([0, 2, 3]), sorted(edges))

        edges = medrecord.edges_connecting("0", "1", directed=False)

        self.assertEqual([0, 1], sorted(edges))

    def test_add_node(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_node("0", {})

        self.assertEqual(1, medrecord.node_count())
        self.assertEqual(0, len(medrecord.groups))

        medrecord = MedRecord()

        medrecord.add_node("0", {}, "0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertEqual(1, len(medrecord.groups))

    def test_invalid_add_node(self):
        medrecord = create_medrecord()

        with self.assertRaises(AssertionError):
            medrecord.add_node("0", {})

    def test_remove_node(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.node_count())

        attributes = medrecord.remove_node("0")

        self.assertEqual(3, medrecord.node_count())
        self.assertEqual(create_nodes()[0][1], attributes)

        attributes = medrecord.remove_node(["1", "2"])

        self.assertEqual(1, medrecord.node_count())
        self.assertEqual(
            {"1": create_nodes()[1][1], "2": create_nodes()[2][1]}, attributes
        )

        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.node_count())

        attributes = medrecord.remove_node(node_select().index().is_in(["0", "1"]))

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(
            {"0": create_nodes()[0][1], "1": create_nodes()[1][1]}, attributes
        )

    def test_invalid_remove_node(self):
        medrecord = create_medrecord()

        # Removing a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node("50")

        # Removing a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node(["0", "50"])

    def test_add_nodes(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(create_nodes())

        self.assertEqual(4, medrecord.node_count())

        # Adding tuple to a group
        medrecord = MedRecord()

        medrecord.add_nodes(create_nodes(), "0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))
        self.assertIn("0", medrecord.groups)

        # Adding group without nodes
        medrecord = MedRecord()

        medrecord.add_nodes([], "0")

        self.assertEqual(0, medrecord.node_count())
        self.assertIn("0", medrecord.groups)

        # Adding pandas dataframe
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes((create_pandas_nodes_dataframe(), "index"))

        self.assertEqual(2, medrecord.node_count())

        # Adding pandas dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes((create_pandas_nodes_dataframe(), "index"), "0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))

        # Adding polars dataframe
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        medrecord.add_nodes((nodes, "index"))

        self.assertEqual(2, medrecord.node_count())

        # Adding polars dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes((nodes, "index"), "0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))

        # Adding multiple pandas dataframes
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ]
        )

        self.assertEqual(4, medrecord.node_count())

        # Adding multiple pandas dataframes to a group
        medrecord = MedRecord()

        medrecord.add_nodes(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
            group="0",
        )

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))

        # Checking if nodes can be added to a group that already exists
        medrecord = MedRecord()

        medrecord.add_nodes((create_pandas_nodes_dataframe(), "index"), group="0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertNotIn("2", medrecord.nodes_in_group("0"))
        self.assertNotIn("3", medrecord.nodes_in_group("0"))

        medrecord.add_nodes(
            (create_second_pandas_nodes_dataframe(), "index"), group="0"
        )

        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))

        # Adding multiple polars dataframes
        medrecord = MedRecord()

        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(
            [
                (nodes, "index"),
                (second_nodes, "index"),
            ]
        )

        self.assertEqual(4, medrecord.node_count())

        # Adding multiple polars dataframes to a group
        medrecord = MedRecord()

        medrecord.add_nodes(
            [
                (nodes, "index"),
                (second_nodes, "index"),
            ],
            group="0",
        )

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))

    def test_invalid_add_nodes(self):
        medrecord = create_medrecord()

        with self.assertRaises(AssertionError):
            medrecord.add_nodes(create_nodes())

    def test_add_nodes_pandas(self):
        medrecord = MedRecord()

        nodes = (create_pandas_nodes_dataframe(), "index")

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes_pandas(nodes)

        self.assertEqual(2, medrecord.node_count())

        medrecord = MedRecord()

        second_nodes = (create_second_pandas_nodes_dataframe(), "index")

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes_pandas([nodes, second_nodes])

        self.assertEqual(4, medrecord.node_count())

        # Trying with the group argument
        medrecord = MedRecord()

        medrecord.add_nodes_pandas(nodes, group="0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))

        medrecord = MedRecord()

        medrecord.add_nodes_pandas([], group="0")

        self.assertEqual(0, medrecord.node_count())
        self.assertIn("0", medrecord.groups)

        medrecord = MedRecord()

        medrecord.add_nodes_pandas([nodes, second_nodes], group="0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))

    def test_add_nodes_polars(self):
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes_polars((nodes, "index"))

        self.assertEqual(2, medrecord.node_count())

        medrecord = MedRecord()

        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes_polars([(nodes, "index"), (second_nodes, "index")])

        self.assertEqual(4, medrecord.node_count())

        # Trying with the group argument
        medrecord = MedRecord()

        medrecord.add_nodes_polars((nodes, "index"), group="0")

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))

        medrecord = MedRecord()

        medrecord.add_nodes_polars([], group="0")

        self.assertEqual(0, medrecord.node_count())
        self.assertIn("0", medrecord.groups)

        medrecord = MedRecord()

        medrecord.add_nodes_polars(
            [(nodes, "index"), (second_nodes, "index")], group="0"
        )

        self.assertIn("0", medrecord.nodes_in_group("0"))
        self.assertIn("1", medrecord.nodes_in_group("0"))
        self.assertIn("2", medrecord.nodes_in_group("0"))
        self.assertIn("3", medrecord.nodes_in_group("0"))

    def test_invalid_add_nodes_polars(self):
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        # Adding a nodes dataframe with the wrong index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_nodes_polars((nodes, "invalid"))

        # Adding a nodes dataframe with the wrong index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_nodes_polars([(nodes, "index"), (second_nodes, "invalid")])

    def test_add_edge(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.edge_count())

        medrecord.add_edge("0", "3", {})

        self.assertEqual(5, medrecord.edge_count())

        medrecord.add_edge("3", "0", {}, group="0")

        self.assertEqual(6, medrecord.edge_count())
        self.assertIn(5, medrecord.edges_in_group("0"))

    def test_invalid_add_edge(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        # Adding an edge pointing to a non-existent node should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge("0", "50", {})

        # Adding an edge from a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge("50", "0", {})

    def test_remove_edge(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.edge_count())

        attributes = medrecord.remove_edge(0)

        self.assertEqual(3, medrecord.edge_count())
        self.assertEqual(create_edges()[0][2], attributes)

        attributes = medrecord.remove_edge([1, 2])

        self.assertEqual(1, medrecord.edge_count())
        self.assertEqual({1: create_edges()[1][2], 2: create_edges()[2][2]}, attributes)

        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.edge_count())

        attributes = medrecord.remove_edge(edge_select().index().is_in([0, 1]))

        self.assertEqual(2, medrecord.edge_count())
        self.assertEqual({0: create_edges()[0][2], 1: create_edges()[1][2]}, attributes)

    def test_invalid_remove_edge(self):
        medrecord = create_medrecord()

        # Removing a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge(50)

    def test_add_edges(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(create_edges())

        self.assertEqual(4, medrecord.edge_count())

        # Adding tuple to a group

        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges(create_edges(), "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))
        self.assertIn(2, medrecord.edges_in_group("0"))
        self.assertIn(3, medrecord.edges_in_group("0"))

        # Adding pandas dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges((create_pandas_edges_dataframe(), "source", "target"))

        self.assertEqual(2, medrecord.edge_count())

        # Adding pandas dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges((create_pandas_edges_dataframe(), "source", "target"), "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))

        # Adding polars dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        medrecord.add_edges((edges, "source", "target"))

        self.assertEqual(2, medrecord.edge_count())

        # Adding polars dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges((edges, "source", "target"), "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))

        # Adding multiple pandas dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ]
        )

        self.assertEqual(4, medrecord.edge_count())

        # Adding multiple pandas dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges(
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ],
            "0",
        )

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))
        self.assertIn(2, medrecord.edges_in_group("0"))
        self.assertIn(3, medrecord.edges_in_group("0"))

        # Adding multiple polars dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        medrecord.add_edges(
            [
                (edges, "source", "target"),
                (second_edges, "source", "target"),
            ]
        )

        self.assertEqual(4, medrecord.edge_count())

        # Adding multiple polars dataframe to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges(
            [
                (edges, "source", "target"),
                (second_edges, "source", "target"),
            ],
            "0",
        )

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))
        self.assertIn(2, medrecord.edges_in_group("0"))
        self.assertIn(3, medrecord.edges_in_group("0"))

    def test_add_edges_pandas(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = (create_pandas_edges_dataframe(), "source", "target")

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(edges)

        self.assertEqual(2, medrecord.edge_count())

        # Adding to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges(edges, "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))

        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        second_edges = (create_second_pandas_edges_dataframe(), "source", "target")

        medrecord.add_edges([edges, second_edges], "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))
        self.assertIn(2, medrecord.edges_in_group("0"))
        self.assertIn(3, medrecord.edges_in_group("0"))

    def test_add_edges_polars(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges_polars((edges, "source", "target"))

        self.assertEqual(2, medrecord.edge_count())

        # Adding to a group
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        medrecord.add_edges_polars((edges, "source", "target"), "0")

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))

        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        medrecord.add_edges_polars(
            [(edges, "source", "target"), (second_edges, "source", "target")], "0"
        )

        self.assertIn(0, medrecord.edges_in_group("0"))
        self.assertIn(1, medrecord.edges_in_group("0"))
        self.assertIn(2, medrecord.edges_in_group("0"))
        self.assertIn(3, medrecord.edges_in_group("0"))

    def test_invalid_add_edges_polars(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        # Providing the wrong source index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_edges_polars((edges, "invalid", "target"))

        # Providing the wrong target index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_edges_polars((edges, "source", "invalid"))

    def test_add_group(self):
        medrecord = create_medrecord()

        self.assertEqual(0, medrecord.group_count())

        medrecord.add_group("0")

        self.assertEqual(1, medrecord.group_count())

        medrecord.add_group("1", "0", 0)

        self.assertEqual(2, medrecord.group_count())
        self.assertEqual({"nodes": ["0"], "edges": [0]}, medrecord.group("1"))

        medrecord.add_group("2", ["0", "1"], [0, 1])

        self.assertEqual(3, medrecord.group_count())
        nodes_and_edges = medrecord.group("2")
        self.assertEqual(sorted(["0", "1"]), sorted(nodes_and_edges["nodes"]))
        self.assertEqual(sorted([0, 1]), sorted(nodes_and_edges["edges"]))

        medrecord.add_group(
            "3",
            node_select().index().is_in(["0", "1"]),
            edge_select().index().is_in([0, 1]),
        )

        self.assertEqual(4, medrecord.group_count())
        nodes_and_edges = medrecord.group("3")
        self.assertEqual(sorted(["0", "1"]), sorted(nodes_and_edges["nodes"]))
        self.assertEqual(sorted([0, 1]), sorted(nodes_and_edges["edges"]))

    def test_invalid_add_group(self):
        medrecord = create_medrecord()

        # Adding a group with a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.add_group("0", "50")

        # Adding an already existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_group("0", ["0", "50"])

        medrecord.add_group("0", "0")

        # Adding an already existing group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_group("0")

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_group("0", "0")

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_group("0", ["1", "0"])

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_group("0", node_select().index() == "0")

    def test_remove_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual(1, medrecord.group_count())

        medrecord.remove_group("0")

        self.assertEqual(0, medrecord.group_count())

    def test_invalid_remove_group(self):
        medrecord = create_medrecord()

        # Removing a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_group("0")

    def test_add_node_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual([], medrecord.nodes_in_group("0"))

        medrecord.add_node_to_group("0", "0")

        self.assertEqual(["0"], medrecord.nodes_in_group("0"))

        medrecord.add_node_to_group("0", ["1", "2"])

        self.assertEqual(
            sorted(["0", "1", "2"]),
            sorted(medrecord.nodes_in_group("0")),
        )

        medrecord.add_node_to_group("0", node_select().index() == "3")

        self.assertEqual(
            sorted(["0", "1", "2", "3"]),
            sorted(medrecord.nodes_in_group("0")),
        )

    def test_invalid_add_node_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0"])

        # Adding to a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_node_to_group("50", "1")

        # Adding to a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_node_to_group("50", ["1", "2"])

        # Adding a non-existing node to a group should fail
        with self.assertRaises(IndexError):
            medrecord.add_node_to_group("0", "50")

        # Adding a non-existing node to a group should fail
        with self.assertRaises(IndexError):
            medrecord.add_node_to_group("0", ["1", "50"])

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_node_to_group("0", "0")

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_node_to_group("0", ["1", "0"])

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_node_to_group("0", node_select().index() == "0")

    def test_add_edge_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual([], medrecord.edges_in_group("0"))

        medrecord.add_edge_to_group("0", 0)

        self.assertEqual([0], medrecord.edges_in_group("0"))

        medrecord.add_edge_to_group("0", [1, 2])

        self.assertEqual(
            sorted([0, 1, 2]),
            sorted(medrecord.edges_in_group("0")),
        )

        medrecord.add_edge_to_group("0", edge_select().index() == 3)

        self.assertEqual(
            sorted([0, 1, 2, 3]),
            sorted(medrecord.edges_in_group("0")),
        )

    def test_invalid_add_edge_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0])

        # Adding to a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge_to_group("50", 1)

        # Adding to a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge_to_group("50", [1, 2])

        # Adding a non-existing edge to a group should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge_to_group("0", 50)

        # Adding a non-existing edge to a group should fail
        with self.assertRaises(IndexError):
            medrecord.add_edge_to_group("0", [1, 50])

        # Adding an edge to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_edge_to_group("0", 0)

        # Adding an edge to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_edge_to_group("0", [1, 0])

        # Adding an edge to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_edge_to_group("0", edge_select().index() == 0)

    def test_remove_node_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        self.assertEqual(
            sorted(["0", "1"]),
            sorted(medrecord.nodes_in_group("0")),
        )

        medrecord.remove_node_from_group("0", "1")

        self.assertEqual(["0"], medrecord.nodes_in_group("0"))

        medrecord.add_node_to_group("0", "1")

        self.assertEqual(
            sorted(["0", "1"]),
            sorted(medrecord.nodes_in_group("0")),
        )

        medrecord.remove_node_from_group("0", ["0", "1"])

        self.assertEqual([], medrecord.nodes_in_group("0"))

        medrecord.add_node_to_group("0", ["0", "1"])

        self.assertEqual(
            sorted(["0", "1"]),
            sorted(medrecord.nodes_in_group("0")),
        )

        medrecord.remove_node_from_group("0", node_select().index().is_in(["0", "1"]))

        self.assertEqual([], medrecord.nodes_in_group("0"))

    def test_invalid_remove_node_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        # Removing a node from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node_from_group("50", "0")

        # Removing a node from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node_from_group("50", ["0", "1"])

        # Removing a node from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node_from_group("50", node_select().index() == "0")

        # Removing a non-existing node from a group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node_from_group("0", "50")

        # Removing a non-existing node from a group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_node_from_group("0", ["0", "50"])

    def test_remove_edge_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.edges_in_group("0")),
        )

        medrecord.remove_edge_from_group("0", 1)

        self.assertEqual([0], medrecord.edges_in_group("0"))

        medrecord.add_edge_to_group("0", 1)

        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.edges_in_group("0")),
        )

        medrecord.remove_edge_from_group("0", [0, 1])

        self.assertEqual([], medrecord.edges_in_group("0"))

        medrecord.add_edge_to_group("0", [0, 1])

        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.edges_in_group("0")),
        )

        medrecord.remove_edge_from_group("0", edge_select().index().is_in([0, 1]))

        self.assertEqual([], medrecord.edges_in_group("0"))

    def test_invalid_remove_edge_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        # Removing an edge from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge_from_group("50", 0)

        # Removing an edge from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge_from_group("50", [0, 1])

        # Removing an edge from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge_from_group("50", edge_select().index() == 0)

        # Removing a non-existing edge from a group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge_from_group("0", 50)

        # Removing a non-existing edge from a group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_edge_from_group("0", [0, 50])

    def test_nodes_in_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        self.assertEqual(
            sorted(["0", "1"]),
            sorted(medrecord.nodes_in_group("0")),
        )

    def test_invalid_nodes_in_group(self):
        medrecord = create_medrecord()

        # Querying nodes in a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.nodes_in_group("50")

    def test_edges_in_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.edges_in_group("0")),
        )

    def test_invalid_edges_in_group(self):
        medrecord = create_medrecord()

        # Querying edges in a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.edges_in_group("50")

    def test_groups_of_node(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        self.assertEqual(["0"], medrecord.groups_of_node("0"))

        self.assertEqual({"0": ["0"], "1": ["0"]}, medrecord.groups_of_node(["0", "1"]))

        self.assertEqual(
            {"0": ["0"], "1": ["0"]},
            medrecord.groups_of_node(node_select().index().is_in(["0", "1"])),
        )

    def test_invalid_groups_of_node(self):
        medrecord = create_medrecord()

        # Querying groups of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.groups_of_node("50")

        # Querying groups of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.groups_of_node(["0", "50"])

    def test_groups_of_edge(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        self.assertEqual(["0"], medrecord.groups_of_edge(0))

        self.assertEqual({0: ["0"], 1: ["0"]}, medrecord.groups_of_edge([0, 1]))

        self.assertEqual(
            {0: ["0"], 1: ["0"]},
            medrecord.groups_of_edge(edge_select().index().is_in([0, 1])),
        )

    def test_invalid_groups_of_edge(self):
        medrecord = create_medrecord()

        # Querying groups of a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.groups_of_edge(50)

        # Querying groups of a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.groups_of_edge([0, 50])

    def test_node_count(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_node("0", {})

        self.assertEqual(1, medrecord.node_count())

    def test_edge_count(self):
        medrecord = MedRecord()

        medrecord.add_node("0", {})
        medrecord.add_node("1", {})

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edge("0", "1", {})

        self.assertEqual(1, medrecord.edge_count())

    def test_group_count(self):
        medrecord = create_medrecord()

        self.assertEqual(0, medrecord.group_count())

        medrecord.add_group("0")

        self.assertEqual(1, medrecord.group_count())

    def test_contains_node(self):
        medrecord = create_medrecord()

        self.assertTrue(medrecord.contains_node("0"))

        self.assertFalse(medrecord.contains_node("50"))

    def test_contains_edge(self):
        medrecord = create_medrecord()

        self.assertTrue(medrecord.contains_edge(0))

        self.assertFalse(medrecord.contains_edge(50))

    def test_contains_group(self):
        medrecord = create_medrecord()

        self.assertFalse(medrecord.contains_group("0"))

        medrecord.add_group("0")

        self.assertTrue(medrecord.contains_group("0"))

    def test_neighbors(self):
        medrecord = create_medrecord()

        neighbors = medrecord.neighbors("0")

        self.assertEqual(
            sorted(["1", "3"]),
            sorted(neighbors),
        )

        neighbors = medrecord.neighbors(["0", "1"])

        self.assertEqual(
            {"0": sorted(["1", "3"]), "1": ["0", "2"]},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

        neighbors = medrecord.neighbors(node_select().index().is_in(["0", "1"]))

        self.assertEqual(
            {"0": sorted(["1", "3"]), "1": ["0", "2"]},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

        neighbors = medrecord.neighbors("0", directed=False)

        self.assertEqual(
            sorted(["1", "3"]),
            sorted(neighbors),
        )

        neighbors = medrecord.neighbors(["0", "1"], directed=False)

        self.assertEqual(
            {"0": sorted(["1", "3"]), "1": ["0", "2"]},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

        neighbors = medrecord.neighbors(
            node_select().index().is_in(["0", "1"]), directed=False
        )

        self.assertEqual(
            {"0": sorted(["1", "3"]), "1": ["0", "2"]},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

    def test_invalid_neighbors(self):
        medrecord = create_medrecord()

        # Querying neighbors of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.neighbors("50")

        # Querying neighbors of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.neighbors(["0", "50"])

        # Querying undirected neighbors of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.neighbors("50", directed=False)

        # Querying undirected neighbors of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.neighbors(["0", "50"], directed=False)

    def test_clear(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(4, medrecord.edge_count())
        self.assertEqual(0, medrecord.group_count())

        medrecord.clear()

        self.assertEqual(0, medrecord.node_count())
        self.assertEqual(0, medrecord.edge_count())
        self.assertEqual(0, medrecord.group_count())

    def test_clone(self):
        medrecord = create_medrecord()

        cloned_medrecord = medrecord.clone()

        self.assertEqual(medrecord.node_count(), cloned_medrecord.node_count())
        self.assertEqual(medrecord.edge_count(), cloned_medrecord.edge_count())
        self.assertEqual(medrecord.group_count(), cloned_medrecord.group_count())

        cloned_medrecord.add_node("new_node", {"attribute": "value"})
        cloned_medrecord.add_edge("0", "new_node", {"attribute": "value"})
        cloned_medrecord.add_group("new_group", ["new_node"])

        self.assertNotEqual(medrecord.node_count(), cloned_medrecord.node_count())
        self.assertNotEqual(medrecord.edge_count(), cloned_medrecord.edge_count())
        self.assertNotEqual(medrecord.group_count(), cloned_medrecord.group_count())

    def test_describe_group_nodes(self):
        medrecord = create_medrecord()

        medrecord.add_group("Float")
        medrecord.add_group(1, nodes=["2", "0"])
        medrecord.add_group("Odd", nodes=["1", "3"])

        self.assertDictEqual(
            medrecord._describe_group_nodes(),
            {
                1: {
                    "count": 2,
                    "attribute": {
                        "adipiscing": ["Values: elit"],
                        "dolor": ["Values: sit"],
                        "lorem": ["Values: ipsum"],
                    },
                },
                "Float": {"count": 0, "attribute": {}},
                "Odd": {"count": 2, "attribute": {"amet": ["Values: consectetur"]}},
            },
        )

    def test_describe_group_edges(self):
        medrecord = create_medrecord()

        medrecord.add_group("Even", edges=[0, 2])

        self.assertDictEqual(
            medrecord._describe_group_edges(),
            {
                "Even": {
                    "count": 2,
                    "attribute": {
                        "eiusmod": ["Values: tempor"],
                        "incididunt": ["Values: ut"],
                        "sed": ["Values: do"],
                    },
                },
                "Ungrouped Nodes": {"count": 2, "attribute": {}},
            },
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMedRecord)
    unittest.TextTestRunner(verbosity=2).run(run_test)

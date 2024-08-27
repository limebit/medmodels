import tempfile
import unittest
from typing import List, Tuple

import pandas as pd
import polars as pl
import pytest

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
    def test_from_tuples(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 4

    def test_invalid_from_tuples(self) -> None:
        nodes = create_nodes()

        # Adding an edge pointing to a non-existent node should fail
        with pytest.raises(IndexError):
            MedRecord.from_tuples(nodes, [("0", "50", {})])

        # Adding an edge from a non-existing node should fail
        with pytest.raises(IndexError):
            MedRecord.from_tuples(nodes, [("50", "0", {})])

    def test_from_pandas(self) -> None:
        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
        )

        assert medrecord.node_count() == 2
        assert medrecord.edge_count() == 0

        medrecord = MedRecord.from_pandas(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
        )

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 0

        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
            (create_pandas_edges_dataframe(), "source", "target"),
        )

        assert medrecord.node_count() == 2
        assert medrecord.edge_count() == 2

        medrecord = MedRecord.from_pandas(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ],
            (create_pandas_edges_dataframe(), "source", "target"),
        )

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 2

        medrecord = MedRecord.from_pandas(
            (create_pandas_nodes_dataframe(), "index"),
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ],
        )

        assert medrecord.node_count() == 2
        assert medrecord.edge_count() == 4

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

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 4

    def test_from_polars(self) -> None:
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())
        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        medrecord = MedRecord.from_polars((nodes, "index"), (edges, "source", "target"))

        assert medrecord.node_count() == 2
        assert medrecord.edge_count() == 2

        medrecord = MedRecord.from_polars(
            [(nodes, "index"), (second_nodes, "index")], (edges, "source", "target")
        )

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 2

        medrecord = MedRecord.from_polars(
            (nodes, "index"),
            [(edges, "source", "target"), (second_edges, "source", "target")],
        )

        assert medrecord.node_count() == 2
        assert medrecord.edge_count() == 4

        medrecord = MedRecord.from_polars(
            [(nodes, "index"), (second_nodes, "index")],
            [(edges, "source", "target"), (second_edges, "source", "target")],
        )

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 4

    def test_invalid_from_polars(self) -> None:
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())
        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        # Providing the wrong node index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars((nodes, "invalid"), (edges, "source", "target"))

        # Providing the wrong node index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars(
                [(nodes, "index"), (second_nodes, "invalid")],
                (edges, "source", "target"),
            )

        # Providing the wrong source index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars((nodes, "index"), (edges, "invalid", "target"))

        # Providing the wrong source index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars(
                (nodes, "index"),
                [(edges, "source", "target"), (second_edges, "invalid", "target")],
            )

        # Providing the wrong target index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars((nodes, "index"), (edges, "source", "invalid"))

        # Providing the wrong target index column name should fail
        with pytest.raises(RuntimeError):
            MedRecord.from_polars(
                (nodes, "index"),
                [(edges, "source", "target"), (edges, "source", "invalid")],
            )

    def test_from_example_dataset(self) -> None:
        medrecord = MedRecord.from_example_dataset()

        assert medrecord.node_count() == 73
        assert medrecord.edge_count() == 160

        assert len(medrecord.nodes_in_group("diagnosis")) == 25
        assert len(medrecord.nodes_in_group("drug")) == 19
        assert len(medrecord.nodes_in_group("patient")) == 5
        assert len(medrecord.nodes_in_group("procedure")) == 24

    def test_ron(self) -> None:
        medrecord = create_medrecord()

        with tempfile.NamedTemporaryFile() as f:
            medrecord.to_ron(f.name)

            loaded_medrecord = MedRecord.from_ron(f.name)

        assert medrecord.node_count() == loaded_medrecord.node_count()
        assert medrecord.edge_count() == loaded_medrecord.edge_count()

    def test_schema(self) -> None:
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

        with pytest.raises(ValueError):
            medrecord.add_node("1", {"attribute": "1"})

        medrecord.add_node("1", {"attribute": 1, "attribute2": 1})

        medrecord.add_node_to_group("group", "1")

        medrecord.add_node("2", {"attribute": 1, "attribute2": "1"})

        with pytest.raises(ValueError):
            medrecord.add_node_to_group("group", "2")

        medrecord.add_edge("0", "1", {"attribute": 1})

        with pytest.raises(ValueError):
            medrecord.add_edge("0", "1", {"attribute": "1"})

        edge_index = medrecord.add_edge("0", "1", {"attribute": 1, "attribute2": 1})

        medrecord.add_edge_to_group("group", edge_index)

        edge_index = medrecord.add_edge("0", "1", {"attribute": 1, "attribute2": "1"})

        with pytest.raises(ValueError):
            medrecord.add_edge_to_group("group", edge_index)

    def test_nodes(self) -> None:
        medrecord = create_medrecord()

        nodes = [x[0] for x in create_nodes()]

        for node in medrecord.nodes:
            assert node in nodes

    def test_edges(self) -> None:
        medrecord = create_medrecord()

        edges = list(range(len(create_edges())))

        for edge in medrecord.edges:
            assert edge in edges

    def test_groups(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0")

        assert medrecord.groups == ["0"]

    def test_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0")

        assert medrecord.group("0") == {"nodes": [], "edges": []}

        medrecord.add_group("1", ["0"], [0])

        assert medrecord.group("1") == {"nodes": ["0"], "edges": [0]}

        assert medrecord.group(["0", "1"]) == {"0": {"nodes": [], "edges": []}, "1": {"nodes": ["0"], "edges": [0]}}

    def test_invalid_group(self) -> None:
        medrecord = create_medrecord()

        # Querying a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.group("0")

        medrecord.add_group("1", ["0"])

        # Querying a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.group(["0", "50"])

    def test_outgoing_edges(self) -> None:
        medrecord = create_medrecord()

        edges = medrecord.outgoing_edges("0")

        assert sorted([0, 3]) == sorted(edges)

        edges = medrecord.outgoing_edges(["0", "1"])

        assert {key: sorted(value) for key, value in edges.items()} == {"0": sorted([0, 3]), "1": [1, 2]}

        edges = medrecord.outgoing_edges(node_select().index().is_in(["0", "1"]))

        assert {key: sorted(value) for key, value in edges.items()} == {"0": sorted([0, 3]), "1": [1, 2]}

    def test_invalid_outgoing_edges(self) -> None:
        medrecord = create_medrecord()

        # Querying outgoing edges of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.outgoing_edges("50")

        # Querying outgoing edges of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.outgoing_edges(["0", "50"])

    def test_incoming_edges(self) -> None:
        medrecord = create_medrecord()

        edges = medrecord.incoming_edges("1")

        assert edges == [0]

        edges = medrecord.incoming_edges(["1", "2"])

        assert edges == {"1": [0], "2": [2]}

        edges = medrecord.incoming_edges(node_select().index().is_in(["1", "2"]))

        assert edges == {"1": [0], "2": [2]}

    def test_invalid_incoming_edges(self) -> None:
        medrecord = create_medrecord()

        # Querying incoming edges of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.incoming_edges("50")

        # Querying incoming edges of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.incoming_edges(["0", "50"])

    def test_edge_endpoints(self) -> None:
        medrecord = create_medrecord()

        endpoints = medrecord.edge_endpoints(0)

        assert endpoints == ("0", "1")

        endpoints = medrecord.edge_endpoints([0, 1])

        assert endpoints == {0: ("0", "1"), 1: ("1", "0")}

        endpoints = medrecord.edge_endpoints(edge_select().index().is_in([0, 1]))

        assert endpoints == {0: ("0", "1"), 1: ("1", "0")}

    def test_invalid_edge_endpoints(self) -> None:
        medrecord = create_medrecord()

        # Querying endpoints of a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.edge_endpoints(50)

        # Querying endpoints of a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.edge_endpoints([0, 50])

    def test_edges_connecting(self) -> None:
        medrecord = create_medrecord()

        edges = medrecord.edges_connecting("0", "1")

        assert edges == [0]

        edges = medrecord.edges_connecting(["0", "1"], "1")

        assert edges == [0]

        edges = medrecord.edges_connecting(node_select().index().is_in(["0", "1"]), "1")

        assert edges == [0]

        edges = medrecord.edges_connecting("0", ["1", "3"])

        assert sorted([0, 3]) == sorted(edges)

        edges = medrecord.edges_connecting("0", node_select().index().is_in(["1", "3"]))

        assert sorted([0, 3]) == sorted(edges)

        edges = medrecord.edges_connecting(["0", "1"], ["1", "2", "3"])

        assert sorted([0, 2, 3]) == sorted(edges)

        edges = medrecord.edges_connecting(
            node_select().index().is_in(["0", "1"]),
            node_select().index().is_in(["1", "2", "3"]),
        )

        assert sorted([0, 2, 3]) == sorted(edges)

        edges = medrecord.edges_connecting("0", "1", directed=False)

        assert sorted(edges) == [0, 1]

    def test_add_node(self) -> None:
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        medrecord.add_node("0", {})

        assert medrecord.node_count() == 1

    def test_invalid_add_node(self) -> None:
        medrecord = create_medrecord()

        with pytest.raises(AssertionError):
            medrecord.add_node("0", {})

    def test_remove_node(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.node_count() == 4

        attributes = medrecord.remove_node("0")

        assert medrecord.node_count() == 3
        assert create_nodes()[0][1] == attributes

        attributes = medrecord.remove_node(["1", "2"])

        assert medrecord.node_count() == 1
        assert attributes == {"1": create_nodes()[1][1], "2": create_nodes()[2][1]}

        medrecord = create_medrecord()

        assert medrecord.node_count() == 4

        attributes = medrecord.remove_node(node_select().index().is_in(["0", "1"]))

        assert medrecord.node_count() == 2
        assert attributes == {"0": create_nodes()[0][1], "1": create_nodes()[1][1]}

    def test_invalid_remove_node(self) -> None:
        medrecord = create_medrecord()

        # Removing a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.remove_node("50")

        # Removing a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.remove_node(["0", "50"])

    def test_add_nodes(self) -> None:
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        medrecord.add_nodes(create_nodes())

        assert medrecord.node_count() == 4

        # Adding pandas dataframe
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        medrecord.add_nodes((create_pandas_nodes_dataframe(), "index"))

        assert medrecord.node_count() == 2

        # Adding polars dataframe
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        medrecord.add_nodes((nodes, "index"))

        assert medrecord.node_count() == 2

        # Adding multiple pandas dataframes
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        medrecord.add_nodes(
            [
                (create_pandas_nodes_dataframe(), "index"),
                (create_second_pandas_nodes_dataframe(), "index"),
            ]
        )

        assert medrecord.node_count() == 4

        # Adding multiple polars dataframes
        medrecord = MedRecord()

        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        assert medrecord.node_count() == 0

        medrecord.add_nodes(
            [
                (nodes, "index"),
                (second_nodes, "index"),
            ]
        )

        assert medrecord.node_count() == 4

    def test_invalid_add_nodes(self) -> None:
        medrecord = create_medrecord()

        with pytest.raises(AssertionError):
            medrecord.add_nodes(create_nodes())

    def test_add_nodes_pandas(self) -> None:
        medrecord = MedRecord()

        nodes = (create_pandas_nodes_dataframe(), "index")

        assert medrecord.node_count() == 0

        medrecord.add_nodes(nodes)

        assert medrecord.node_count() == 2

        medrecord = MedRecord()

        second_nodes = (create_second_pandas_nodes_dataframe(), "index")

        assert medrecord.node_count() == 0

        medrecord.add_nodes([nodes, second_nodes])

        assert medrecord.node_count() == 4

    def test_add_nodes_polars(self) -> None:
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        assert medrecord.node_count() == 0

        medrecord.add_nodes_polars((nodes, "index"))

        assert medrecord.node_count() == 2

        medrecord = MedRecord()

        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        assert medrecord.node_count() == 0

        medrecord.add_nodes_polars([(nodes, "index"), (second_nodes, "index")])

        assert medrecord.node_count() == 4

    def test_invalid_add_nodes_polars(self) -> None:
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        second_nodes = pl.from_pandas(create_second_pandas_nodes_dataframe())

        # Adding a nodes dataframe with the wrong index column name should fail
        with pytest.raises(RuntimeError):
            medrecord.add_nodes_polars((nodes, "invalid"))

        # Adding a nodes dataframe with the wrong index column name should fail
        with pytest.raises(RuntimeError):
            medrecord.add_nodes_polars([(nodes, "index"), (second_nodes, "invalid")])

    def test_add_edge(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.edge_count() == 4

        medrecord.add_edge("0", "3", {})

        assert medrecord.edge_count() == 5

    def test_invalid_add_edge(self) -> None:
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        # Adding an edge pointing to a non-existent node should fail
        with pytest.raises(IndexError):
            medrecord.add_edge("0", "50", {})

        # Adding an edge from a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.add_edge("50", "0", {})

    def test_remove_edge(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.edge_count() == 4

        attributes = medrecord.remove_edge(0)

        assert medrecord.edge_count() == 3
        assert create_edges()[0][2] == attributes

        attributes = medrecord.remove_edge([1, 2])

        assert medrecord.edge_count() == 1
        assert attributes == {1: create_edges()[1][2], 2: create_edges()[2][2]}

        medrecord = create_medrecord()

        assert medrecord.edge_count() == 4

        attributes = medrecord.remove_edge(edge_select().index().is_in([0, 1]))

        assert medrecord.edge_count() == 2
        assert attributes == {0: create_edges()[0][2], 1: create_edges()[1][2]}

    def test_invalid_remove_edge(self) -> None:
        medrecord = create_medrecord()

        # Removing a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge(50)

    def test_add_edges(self) -> None:
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        assert medrecord.edge_count() == 0

        medrecord.add_edges(create_edges())

        assert medrecord.edge_count() == 4

        # Adding pandas dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        assert medrecord.edge_count() == 0

        medrecord.add_edges((create_pandas_edges_dataframe(), "source", "target"))

        assert medrecord.edge_count() == 2

        # Adding polars dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        assert medrecord.edge_count() == 0

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        medrecord.add_edges((edges, "source", "target"))

        assert medrecord.edge_count() == 2

        # Adding multiple pandas dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        assert medrecord.edge_count() == 0

        medrecord.add_edges(
            [
                (create_pandas_edges_dataframe(), "source", "target"),
                (create_second_pandas_edges_dataframe(), "source", "target"),
            ]
        )

        assert medrecord.edge_count() == 4

        # Adding multiple polats dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        assert medrecord.edge_count() == 0

        second_edges = pl.from_pandas(create_second_pandas_edges_dataframe())

        medrecord.add_edges(
            [
                (edges, "source", "target"),
                (second_edges, "source", "target"),
            ]
        )

        assert medrecord.edge_count() == 4

    def test_add_edges_pandas(self) -> None:
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = (create_pandas_edges_dataframe(), "source", "target")

        assert medrecord.edge_count() == 0

        medrecord.add_edges(edges)

        assert medrecord.edge_count() == 2

    def test_add_edges_polars(self) -> None:
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        assert medrecord.edge_count() == 0

        medrecord.add_edges_polars((edges, "source", "target"))

        assert medrecord.edge_count() == 2

    def test_invalid_add_edges_polars(self) -> None:
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        # Providing the wrong source index column name should fail
        with pytest.raises(RuntimeError):
            medrecord.add_edges_polars((edges, "invalid", "target"))

        # Providing the wrong target index column name should fail
        with pytest.raises(RuntimeError):
            medrecord.add_edges_polars((edges, "source", "invalid"))

    def test_add_group(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.group_count() == 0

        medrecord.add_group("0")

        assert medrecord.group_count() == 1

        medrecord.add_group("1", "0", 0)

        assert medrecord.group_count() == 2
        assert medrecord.group("1") == {"nodes": ["0"], "edges": [0]}

        medrecord.add_group("2", ["0", "1"], [0, 1])

        assert medrecord.group_count() == 3
        nodes_and_edges = medrecord.group("2")
        assert sorted(["0", "1"]) == sorted(nodes_and_edges["nodes"])
        assert sorted([0, 1]) == sorted(nodes_and_edges["edges"])

        medrecord.add_group(
            "3",
            node_select().index().is_in(["0", "1"]),
            edge_select().index().is_in([0, 1]),
        )

        assert medrecord.group_count() == 4
        nodes_and_edges = medrecord.group("3")
        assert sorted(["0", "1"]) == sorted(nodes_and_edges["nodes"])
        assert sorted([0, 1]) == sorted(nodes_and_edges["edges"])

    def test_invalid_add_group(self) -> None:
        medrecord = create_medrecord()

        # Adding a group with a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.add_group("0", "50")

        # Adding an already existing group should fail
        with pytest.raises(IndexError):
            medrecord.add_group("0", ["0", "50"])

        medrecord.add_group("0", "0")

        # Adding an already existing group should fail
        with pytest.raises(AssertionError):
            medrecord.add_group("0")

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_group("0", "0")

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_group("0", ["1", "0"])

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_group("0", node_select().index() == "0")

    def test_remove_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0")

        assert medrecord.group_count() == 1

        medrecord.remove_group("0")

        assert medrecord.group_count() == 0

    def test_invalid_remove_group(self) -> None:
        medrecord = create_medrecord()

        # Removing a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_group("0")

    def test_add_node_to_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0")

        assert medrecord.nodes_in_group("0") == []

        medrecord.add_node_to_group("0", "0")

        assert medrecord.nodes_in_group("0") == ["0"]

        medrecord.add_node_to_group("0", ["1", "2"])

        assert sorted(["0", "1", "2"]) == sorted(medrecord.nodes_in_group("0"))

        medrecord.add_node_to_group("0", node_select().index() == "3")

        assert sorted(["0", "1", "2", "3"]) == sorted(medrecord.nodes_in_group("0"))

    def test_invalid_add_node_to_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0"])

        # Adding to a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.add_node_to_group("50", "1")

        # Adding to a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.add_node_to_group("50", ["1", "2"])

        # Adding a non-existing node to a group should fail
        with pytest.raises(IndexError):
            medrecord.add_node_to_group("0", "50")

        # Adding a non-existing node to a group should fail
        with pytest.raises(IndexError):
            medrecord.add_node_to_group("0", ["1", "50"])

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_node_to_group("0", "0")

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_node_to_group("0", ["1", "0"])

        # Adding a node to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_node_to_group("0", node_select().index() == "0")

    def test_add_edge_to_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0")

        assert medrecord.edges_in_group("0") == []

        medrecord.add_edge_to_group("0", 0)

        assert medrecord.edges_in_group("0") == [0]

        medrecord.add_edge_to_group("0", [1, 2])

        assert sorted([0, 1, 2]) == sorted(medrecord.edges_in_group("0"))

        medrecord.add_edge_to_group("0", edge_select().index() == 3)

        assert sorted([0, 1, 2, 3]) == sorted(medrecord.edges_in_group("0"))

    def test_invalid_add_edge_to_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0])

        # Adding to a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.add_edge_to_group("50", 1)

        # Adding to a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.add_edge_to_group("50", [1, 2])

        # Adding a non-existing edge to a group should fail
        with pytest.raises(IndexError):
            medrecord.add_edge_to_group("0", 50)

        # Adding a non-existing edge to a group should fail
        with pytest.raises(IndexError):
            medrecord.add_edge_to_group("0", [1, 50])

        # Adding an edge to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_edge_to_group("0", 0)

        # Adding an edge to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_edge_to_group("0", [1, 0])

        # Adding an edge to a group that already is in the group should fail
        with pytest.raises(AssertionError):
            medrecord.add_edge_to_group("0", edge_select().index() == 0)

    def test_remove_node_from_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        assert sorted(["0", "1"]) == sorted(medrecord.nodes_in_group("0"))

        medrecord.remove_node_from_group("0", "1")

        assert medrecord.nodes_in_group("0") == ["0"]

        medrecord.add_node_to_group("0", "1")

        assert sorted(["0", "1"]) == sorted(medrecord.nodes_in_group("0"))

        medrecord.remove_node_from_group("0", ["0", "1"])

        assert medrecord.nodes_in_group("0") == []

        medrecord.add_node_to_group("0", ["0", "1"])

        assert sorted(["0", "1"]) == sorted(medrecord.nodes_in_group("0"))

        medrecord.remove_node_from_group("0", node_select().index().is_in(["0", "1"]))

        assert medrecord.nodes_in_group("0") == []

    def test_invalid_remove_node_from_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        # Removing a node from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_node_from_group("50", "0")

        # Removing a node from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_node_from_group("50", ["0", "1"])

        # Removing a node from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_node_from_group("50", node_select().index() == "0")

        # Removing a non-existing node from a group should fail
        with pytest.raises(IndexError):
            medrecord.remove_node_from_group("0", "50")

        # Removing a non-existing node from a group should fail
        with pytest.raises(IndexError):
            medrecord.remove_node_from_group("0", ["0", "50"])

    def test_remove_edge_from_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        assert sorted([0, 1]) == sorted(medrecord.edges_in_group("0"))

        medrecord.remove_edge_from_group("0", 1)

        assert medrecord.edges_in_group("0") == [0]

        medrecord.add_edge_to_group("0", 1)

        assert sorted([0, 1]) == sorted(medrecord.edges_in_group("0"))

        medrecord.remove_edge_from_group("0", [0, 1])

        assert medrecord.edges_in_group("0") == []

        medrecord.add_edge_to_group("0", [0, 1])

        assert sorted([0, 1]) == sorted(medrecord.edges_in_group("0"))

        medrecord.remove_edge_from_group("0", edge_select().index().is_in([0, 1]))

        assert medrecord.edges_in_group("0") == []

    def test_invalid_remove_edge_from_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        # Removing an edge from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge_from_group("50", 0)

        # Removing an edge from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge_from_group("50", [0, 1])

        # Removing an edge from a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge_from_group("50", edge_select().index() == 0)

        # Removing a non-existing edge from a group should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge_from_group("0", 50)

        # Removing a non-existing edge from a group should fail
        with pytest.raises(IndexError):
            medrecord.remove_edge_from_group("0", [0, 50])

    def test_nodes_in_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        assert sorted(["0", "1"]) == sorted(medrecord.nodes_in_group("0"))

    def test_invalid_nodes_in_group(self) -> None:
        medrecord = create_medrecord()

        # Querying nodes in a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.nodes_in_group("50")

    def test_edges_in_group(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        assert sorted([0, 1]) == sorted(medrecord.edges_in_group("0"))

    def test_invalid_edges_in_group(self) -> None:
        medrecord = create_medrecord()

        # Querying edges in a non-existing group should fail
        with pytest.raises(IndexError):
            medrecord.edges_in_group("50")

    def test_groups_of_node(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        assert medrecord.groups_of_node("0") == ["0"]

        assert medrecord.groups_of_node(["0", "1"]) == {"0": ["0"], "1": ["0"]}

        assert medrecord.groups_of_node(node_select().index().is_in(["0", "1"])) == {"0": ["0"], "1": ["0"]}

    def test_invalid_groups_of_node(self) -> None:
        medrecord = create_medrecord()

        # Querying groups of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.groups_of_node("50")

        # Querying groups of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.groups_of_node(["0", "50"])

    def test_groups_of_edge(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("0", edges=[0, 1])

        assert medrecord.groups_of_edge(0) == ["0"]

        assert medrecord.groups_of_edge([0, 1]) == {0: ["0"], 1: ["0"]}

        assert medrecord.groups_of_edge(edge_select().index().is_in([0, 1])) == {0: ["0"], 1: ["0"]}

    def test_invalid_groups_of_edge(self) -> None:
        medrecord = create_medrecord()

        # Querying groups of a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.groups_of_edge(50)

        # Querying groups of a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.groups_of_edge([0, 50])

    def test_node_count(self) -> None:
        medrecord = MedRecord()

        assert medrecord.node_count() == 0

        medrecord.add_node("0", {})

        assert medrecord.node_count() == 1

    def test_edge_count(self) -> None:
        medrecord = MedRecord()

        medrecord.add_node("0", {})
        medrecord.add_node("1", {})

        assert medrecord.edge_count() == 0

        medrecord.add_edge("0", "1", {})

        assert medrecord.edge_count() == 1

    def test_group_count(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.group_count() == 0

        medrecord.add_group("0")

        assert medrecord.group_count() == 1

    def test_contains_node(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.contains_node("0")

        assert not medrecord.contains_node("50")

    def test_contains_edge(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.contains_edge(0)

        assert not medrecord.contains_edge(50)

    def test_contains_group(self) -> None:
        medrecord = create_medrecord()

        assert not medrecord.contains_group("0")

        medrecord.add_group("0")

        assert medrecord.contains_group("0")

    def test_neighbors(self) -> None:
        medrecord = create_medrecord()

        neighbors = medrecord.neighbors("0")

        assert sorted(["1", "3"]) == sorted(neighbors)

        neighbors = medrecord.neighbors(["0", "1"])

        assert {key: sorted(value) for key, value in neighbors.items()} == {"0": sorted(["1", "3"]), "1": ["0", "2"]}

        neighbors = medrecord.neighbors(node_select().index().is_in(["0", "1"]))

        assert {key: sorted(value) for key, value in neighbors.items()} == {"0": sorted(["1", "3"]), "1": ["0", "2"]}

        neighbors = medrecord.neighbors("0", directed=False)

        assert sorted(["1", "3"]) == sorted(neighbors)

        neighbors = medrecord.neighbors(["0", "1"], directed=False)

        assert {key: sorted(value) for key, value in neighbors.items()} == {"0": sorted(["1", "3"]), "1": ["0", "2"]}

        neighbors = medrecord.neighbors(
            node_select().index().is_in(["0", "1"]), directed=False
        )

        assert {key: sorted(value) for key, value in neighbors.items()} == {"0": sorted(["1", "3"]), "1": ["0", "2"]}

    def test_invalid_neighbors(self) -> None:
        medrecord = create_medrecord()

        # Querying neighbors of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.neighbors("50")

        # Querying neighbors of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.neighbors(["0", "50"])

        # Querying undirected neighbors of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.neighbors("50", directed=False)

        # Querying undirected neighbors of a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.neighbors(["0", "50"], directed=False)

    def test_clear(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.node_count() == 4
        assert medrecord.edge_count() == 4
        assert medrecord.group_count() == 0

        medrecord.clear()

        assert medrecord.node_count() == 0
        assert medrecord.edge_count() == 0
        assert medrecord.group_count() == 0


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMedRecord)
    unittest.TextTestRunner(verbosity=2).run(run_test)

import unittest

import polars as pl
import pandas as pd

from medmodels import MedRecord


def create_nodes():
    return [
        ("0", {"lorem": "ipsum", "dolor": "sit"}),
        ("1", {"amet": "consectetur"}),
        ("2", {"adipiscing": "elit"}),
        ("3", {}),
    ]


def create_edges():
    return [
        ("0", "1", {"sed": "do", "eiusmod": "tempor"}),
        ("1", "2", {"incididunt": "ut"}),
        ("0", "3", {}),
    ]


def create_pandas_nodes_dataframe():
    return pd.DataFrame(
        {
            "index": ["0", "1"],
            "attribute": [1, 2],
        }
    )


def create_pandas_edges_dataframe():
    return pd.DataFrame(
        {
            "from": ["0", "1"],
            "to": ["1", "0"],
            "attribute": [1, 2],
        }
    )


def create_pandas_nodes_dataframe_with_index():
    return create_pandas_nodes_dataframe().set_index("index")


def create_pandas_edges_dataframe_with_index():
    return create_pandas_edges_dataframe().set_index(["from", "to"])


def create_medrecord():
    return MedRecord.from_tuples(create_nodes(), create_edges())


class TestMedRecord(unittest.TestCase):
    def test_from_tuples(self):
        medrecord = create_medrecord()

        self.assertEqual(4, medrecord.node_count())
        self.assertEqual(3, medrecord.edge_count())

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
            create_pandas_nodes_dataframe_with_index(),
            create_pandas_edges_dataframe_with_index(),
        )

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

    def test_invalid_from_pandas(self):
        nodes = create_pandas_nodes_dataframe()
        nodes_with_index = create_pandas_nodes_dataframe_with_index()
        edges = create_pandas_edges_dataframe()
        edges_with_index = create_pandas_edges_dataframe_with_index()

        # Creating a MedRecord from a node DataFrame without an index should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_pandas(nodes, edges_with_index)

        # Creating a MedRecord from an edge DataFrame without an index should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_pandas(nodes_with_index, edges)

        # Creating a MedRecord from an edge DataFrame without 2 level index should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_pandas(nodes_with_index, edges.set_index("from"))

        # Creating a MedRecord from an edge DataFrame with more than 2 level index
        # should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_pandas(
                nodes_with_index, edges.set_index(["from", "to", "attribute"])
            )

    def test_from_polars(self):
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())

        medrecord = MedRecord.from_polars(nodes, "index", edges, "from", "to")

        self.assertEqual(2, medrecord.node_count())
        self.assertEqual(2, medrecord.edge_count())

    def test_invalid_from_polars(self):
        nodes = pl.from_pandas(create_pandas_nodes_dataframe())
        edges = pl.from_pandas(create_pandas_edges_dataframe())

        # Creating a MedRecord from nodes and edges without edge column names
        # should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_polars(nodes, "index", edges)

        # Creating a MedRecord from nodes and edges without edge to column name
        # should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_polars(nodes, "index", edges, "from")

        # Creating a MedRecord from nodes and edges without edge from column name
        # should fail
        with self.assertRaises(AssertionError):
            MedRecord.from_polars(nodes, "index", edges, None, "to")

        # Providing the wrong node index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(nodes, "invalid", edges, "from", "to")

        # Providing the wrong from index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(nodes, "index", edges, "invalid", "to")

        # Providing the wrong to index column name should fail
        with self.assertRaises(RuntimeError):
            MedRecord.from_polars(nodes, "index", edges, "from", "invalid")

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

    def test_nodes(self):
        medrecord = create_medrecord()

        nodes = [x[0] for x in create_nodes()]

        for node in medrecord.nodes:
            self.assertTrue(node in nodes)

    def test_node(self):
        medrecord = create_medrecord()

        nodes = medrecord.node("0")

        self.assertEqual({"0": create_nodes()[0][1]}, nodes)

        nodes = medrecord.node("0", "1")

        self.assertEqual({"0": create_nodes()[0][1], "1": create_nodes()[1][1]}, nodes)

    def test_invalid_node(self):
        medrecord = create_medrecord()

        # Querying a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.node("50")

    def test_edges(self):
        medrecord = create_medrecord()

        edges = list(range(len(create_edges())))

        for edge in medrecord.edges:
            self.assertTrue(edge in edges)

    def test_edge(self):
        medrecord = create_medrecord()

        edges = medrecord.edge(0)

        self.assertEqual({0: create_edges()[0][2]}, edges)

        edges = medrecord.edge(0, 1)

        self.assertEqual({0: create_edges()[0][2], 1: create_edges()[1][2]}, edges)

    def test_invalid_edge(self):
        medrecord = create_medrecord()

        # Querying a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.edge(50)

    def test_edges_connecting(self):
        medrecord = create_medrecord()

        edges = medrecord.edges_connecting("0", "1")

        self.assertEqual(1, len(edges))

    def test_groups(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual(["0"], medrecord.groups)

    def test_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual({"0": []}, medrecord.group("0"))

        medrecord.add_group("1", ["0"])

        self.assertEqual({"1": ["0"]}, medrecord.group("1"))

    def test_invalid_group(self):
        medrecord = create_medrecord()

        # Querying a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.group("0")

    def test_add_node(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_node("0", {})

        self.assertEqual(1, medrecord.node_count())

    def test_add_nodes(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(create_nodes())

        self.assertEqual(4, medrecord.node_count())

        # Adding pandas dataframe
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(create_pandas_nodes_dataframe_with_index())

        self.assertEqual(2, medrecord.node_count())

    def tests_add_nodes_pandas(self):
        medrecord = MedRecord()

        nodes = create_pandas_nodes_dataframe_with_index()

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes(nodes)

        self.assertEqual(2, medrecord.node_count())

    def test_invalid_add_nodes_pandas(self):
        medrecord = MedRecord()

        nodes = create_pandas_nodes_dataframe()

        # Adding a nodes dataframe without an index should fail
        with self.assertRaises(AssertionError):
            medrecord.add_nodes(nodes)

    def test_add_nodes_polars(self):
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        self.assertEqual(0, medrecord.node_count())

        medrecord.add_nodes_polars(nodes, "index")

        self.assertEqual(2, medrecord.node_count())

    def test_invalid_add_nodes_polars(self):
        medrecord = MedRecord()

        nodes = pl.from_pandas(create_pandas_nodes_dataframe())

        # Adding a nodes dataframe with the wrong index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_nodes_polars(nodes, "invalid")

    def test_add_edge(self):
        medrecord = create_medrecord()

        self.assertEqual(3, medrecord.edge_count())

        medrecord.add_edge("0", "3", {})

        self.assertEqual(4, medrecord.edge_count())

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

    def test_add_edges(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(create_edges())

        self.assertEqual(3, medrecord.edge_count())

        # Adding pandas dataframe
        medrecord = MedRecord()

        medrecord.add_nodes(nodes)

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(create_pandas_edges_dataframe_with_index())

        self.assertEqual(2, medrecord.edge_count())

    def test_add_edges_pandas(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = create_pandas_edges_dataframe_with_index()

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges(edges)

        self.assertEqual(2, medrecord.edge_count())

    def test_invalid_add_edges_pandas(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = create_pandas_edges_dataframe()

        # Adding an edges dataframe without an index should fail
        with self.assertRaises(AssertionError):
            medrecord.add_edges(edges)

        # Adding an edges dataframe without a 2-level index should fail
        with self.assertRaises(AssertionError):
            medrecord.add_edges(edges.set_index("from"))

    def test_add_edges_polars(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        self.assertEqual(0, medrecord.edge_count())

        medrecord.add_edges_polars(edges, "from", "to")

        self.assertEqual(2, medrecord.edge_count())

    def test_invalid_add_edges_polars(self):
        medrecord = MedRecord()

        nodes = create_nodes()

        medrecord.add_nodes(nodes)

        edges = pl.from_pandas(create_pandas_edges_dataframe())

        # Providing the wrong from index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_edges_polars(edges, "invalid", "to")

        # Providing the wrong to index column name should fail
        with self.assertRaises(RuntimeError):
            medrecord.add_edges_polars(edges, "from", "invalid")

    def test_add_group(self):
        medrecord = MedRecord()

        self.assertEqual(0, medrecord.group_count())

        medrecord.add_group("0")

        self.assertEqual(1, medrecord.group_count())

    def test_invalid_add_group(self):
        medrecord = MedRecord()

        # Adding a group with a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.add_group("0", ["50"])

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

    def test_remove_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0", "1"])

        self.assertEqual({"0": ["0", "1"]}, medrecord.group("0"))

        medrecord.remove_from_group("0", "1")

        self.assertEqual({"0": ["0"]}, medrecord.group("0"))

    def test_invalid_remove_from_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0"])

        # Removing a node from a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_from_group("50", "0")

        # Removing a non-existing node from a group should fail
        with self.assertRaises(IndexError):
            medrecord.remove_from_group("0", "50")

    def test_add_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0")

        self.assertEqual({"0": []}, medrecord.group("0"))

        medrecord.add_to_group("0", "0")

        self.assertEqual({"0": ["0"]}, medrecord.group("0"))

        medrecord.add_to_group("0", "1", "2")

        self.assertEqual({"0": ["0", "1", "2"]}, medrecord.group("0"))

    def test_invalid_add_to_group(self):
        medrecord = create_medrecord()

        medrecord.add_group("0", ["0"])

        # Adding to a non-existing group should fail
        with self.assertRaises(IndexError):
            medrecord.add_to_group("50", "0")

        # Adding a non-existing node to a group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_to_group("0", "50")

        # Adding a node to a group that already is in the group should fail
        with self.assertRaises(AssertionError):
            medrecord.add_to_group("0", "0")

    def test_neighbors(self):
        medrecord = create_medrecord()

        neighbors = medrecord.neighbors("0")

        self.assertEqual(
            {"0": sorted(["1", "3"])},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

        neighbors = medrecord.neighbors("0", "1")

        self.assertEqual(
            {"0": sorted(["1", "3"]), "1": ["2"]},
            {key: sorted(value) for (key, value) in neighbors.items()},
        )

    def test_invalid_neighbors(self):
        medrecord = MedRecord()

        # Querying neighbors of a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.neighbors("0")


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMedRecord)
    unittest.TextTestRunner(verbosity=2).run(run_test)

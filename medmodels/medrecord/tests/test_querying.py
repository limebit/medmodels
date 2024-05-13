import unittest
from typing import List, Tuple

from medmodels import MedRecord
from medmodels.medrecord import edge, node
from medmodels.medrecord.types import Attributes, NodeIndex


def create_nodes() -> List[Tuple[NodeIndex, Attributes]]:
    return [
        (
            "0",
            {
                "lorem": "ipsum",
                "dolor": "  ipsum  ",
                "test": "Ipsum",
                "integer": 1,
                "float": 0.5,
            },
        ),
        ("1", {"amet": "consectetur"}),
        ("2", {"adipiscing": "elit"}),
        ("3", {}),
    ]


def create_edges() -> List[Tuple[NodeIndex, NodeIndex, Attributes]]:
    return [
        ("0", "1", {"sed": "do", "eiusmod": "tempor", "dolor": "  do  ", "test": "DO"}),
        ("1", "2", {"incididunt": "ut"}),
        ("0", "2", {"test": 1, "integer": 1, "float": 0.5}),
        ("0", "2", {"test": 0}),
    ]


def create_medrecord() -> MedRecord:
    return MedRecord.from_tuples(create_nodes(), create_edges())


class TestMedRecord(unittest.TestCase):
    def test_select_nodes_node(self):
        medrecord = create_medrecord()

        medrecord.add_group("test", ["0"])

        # Node in group
        self.assertEqual(["0"], medrecord.select_nodes(node().in_group("test")))

        # Node has attribute
        self.assertEqual(["0"], medrecord.select_nodes(node().has_attribute("lorem")))

        # Node has outgoing edge with
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().has_outgoing_edge_with(edge().index().equal(0))
            ),
        )

        # Node has incoming edge with
        self.assertEqual(
            ["1"],
            medrecord.select_nodes(
                node().has_incoming_edge_with(edge().index().equal(0))
            ),
        )

        # Node has edge with
        self.assertEqual(
            sorted(["0", "1"]),
            sorted(
                medrecord.select_nodes(node().has_edge_with(edge().index().equal(0)))
            ),
        )

        # Node has neighbor with
        self.assertEqual(
            sorted(["0", "1"]),
            sorted(
                medrecord.select_nodes(
                    node().has_neighbor_with(node().index().equal("2"))
                )
            ),
        )

    def test_select_nodes_node_index(self):
        medrecord = create_medrecord()

        # Index greater
        self.assertEqual(
            sorted(["2", "3"]),
            sorted(medrecord.select_nodes(node().index().greater("1"))),
        )

        # Index less
        self.assertEqual(
            sorted(["0", "1"]), sorted(medrecord.select_nodes(node().index().less("2")))
        )

        # Index greater or equal
        self.assertEqual(
            sorted(["1", "2", "3"]),
            sorted(medrecord.select_nodes(node().index().greater_or_equal("1"))),
        )

        # Index less or equal
        self.assertEqual(
            sorted(["0", "1", "2"]),
            sorted(medrecord.select_nodes(node().index().less_or_equal("2"))),
        )

        # Index equal
        self.assertEqual(["1"], medrecord.select_nodes(node().index().equal("1")))

        # Index not equal
        self.assertEqual(
            sorted(["0", "2", "3"]),
            sorted(medrecord.select_nodes(node().index().not_equal("1"))),
        )

        # Index in
        self.assertEqual(["1"], medrecord.select_nodes(node().index().is_in(["1"])))

        # Index not in
        self.assertEqual(
            sorted(["0", "2", "3"]),
            sorted(medrecord.select_nodes(node().index().not_in(["1"]))),
        )

        # Index starts with
        self.assertEqual(
            ["1"],
            medrecord.select_nodes(node().index().starts_with("1")),
        )

        # Index ends with
        self.assertEqual(
            ["1"],
            medrecord.select_nodes(node().index().ends_with("1")),
        )

        # Index contains
        self.assertEqual(
            ["1"],
            medrecord.select_nodes(node().index().contains("1")),
        )

    def test_select_nodes_node_attribute(self):
        medrecord = create_medrecord()

        # Attribute greater
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem").greater("ipsum"))
        )
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem") > "ipsum")
        )

        # Attribute less
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem").less("ipsum"))
        )
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem") < "ipsum")
        )

        # Attribute greater or equal
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(node().attribute("lorem").greater_or_equal("ipsum")),
        )
        self.assertEqual(
            ["0"], medrecord.select_nodes(node().attribute("lorem") >= "ipsum")
        )

        # Attribute less or equal
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(node().attribute("lorem").less_or_equal("ipsum")),
        )
        self.assertEqual(
            ["0"], medrecord.select_nodes(node().attribute("lorem") <= "ipsum")
        )

        # Attribute equal
        self.assertEqual(
            ["0"], medrecord.select_nodes(node().attribute("lorem").equal("ipsum"))
        )
        self.assertEqual(
            ["0"], medrecord.select_nodes(node().attribute("lorem") == "ipsum")
        )

        # Attribute not equal
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem").not_equal("ipsum"))
        )
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem") != "ipsum")
        )

        # Attribute in
        self.assertEqual(
            ["0"], medrecord.select_nodes(node().attribute("lorem").is_in(["ipsum"]))
        )

        # Attribute not in
        self.assertEqual(
            [], medrecord.select_nodes(node().attribute("lorem").not_in(["ipsum"]))
        )

        # Attribute starts with
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(node().attribute("lorem").starts_with("ip")),
        )

        # Attribute ends with
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(node().attribute("lorem").ends_with("um")),
        )

        # Attribute contains
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(node().attribute("lorem").contains("su")),
        )

        # Attribute compare to attribute
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem"))
            ),
        )

        # Attribute compare to attribute add
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").add("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") + "10"
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").add("10"))
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") + "10"
            ),
        )

        # Attribute compare to attribute sub
        # Returns nothing because can't sub a string
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").sub("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") + "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").sub("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") - "10"
            ),
        )

        # Attribute compare to attribute sub
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").sub(10))
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node()
                .attribute("integer")
                .not_equal(node().attribute("integer").sub(10))
            ),
        )

        # Attribute compare to attribute mul
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").mul(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") * 2
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").mul(2))
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") * 2
            ),
        )

        # Attribute compare to attribute div
        # Returns nothing because can't div a string
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").div("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") / "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").div("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") / "10"
            ),
        )

        # Attribute compare to attribute div
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").div(2))
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node()
                .attribute("integer")
                .not_equal(node().attribute("integer").div(2))
            ),
        )

        # Attribute compare to attribute pow
        # Returns nothing because can't pow a string
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").pow("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") ** "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").pow("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") ** "10"
            ),
        )

        # Attribute compare to attribute pow
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").pow(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node()
                .attribute("integer")
                .not_equal(node().attribute("integer").pow(2))
            ),
        )

        # Attribute compare to attribute mod
        # Returns nothing because can't mod a string
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").mod("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") == node().attribute("lorem") % "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").mod("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem") != node().attribute("lorem") % "10"
            ),
        )

        # Attribute compare to attribute mod
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").mod(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node()
                .attribute("integer")
                .not_equal(node().attribute("integer").mod(2))
            ),
        )

        # Attribute compare to attribute round
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("lorem").round())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").not_equal(node().attribute("lorem").round())
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("float").round())
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("float").not_equal(node().attribute("float").round())
            ),
        )

        # Attribute compare to attribute round
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("float").ceil())
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("float").not_equal(node().attribute("float").ceil())
            ),
        )

        # Attribute compare to attribute floor
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("float").floor())
            ),
        )
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("float").not_equal(node().attribute("float").floor())
            ),
        )

        # Attribute compare to attribute abs
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").abs())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("integer").not_equal(node().attribute("integer").abs())
            ),
        )

        # Attribute compare to attribute sqrt
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("integer").equal(node().attribute("integer").sqrt())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node()
                .attribute("integer")
                .not_equal(node().attribute("integer").sqrt())
            ),
        )

        # Attribute compare to attribute trim
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("dolor").trim())
            ),
        )

        # Attribute compare to attribute trim_start
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("dolor").trim_start())
            ),
        )

        # Attribute compare to attribute trim_end
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("dolor").trim_end())
            ),
        )

        # Attribute compare to attribute lowercase
        self.assertEqual(
            ["0"],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("test").lowercase())
            ),
        )

        # Attribute compare to attribute uppercase
        self.assertEqual(
            [],
            medrecord.select_nodes(
                node().attribute("lorem").equal(node().attribute("test").uppercase())
            ),
        )

    def test_select_edges_edge(self):
        medrecord = create_medrecord()

        # Edge connected to target
        self.assertEqual(
            sorted([1, 2, 3]),
            sorted(medrecord.select_edges(edge().connected_target("2"))),
        )

        # Edge connected to source
        self.assertEqual(
            sorted([0, 2, 3]),
            sorted(medrecord.select_edges(edge().connected_source("0"))),
        )

        # Edge connected
        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.select_edges(edge().connected("1"))),
        )

        # Edge has attribute
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().has_attribute("sed")),
        )

        # Edge connected to target with
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().connected_target_with(node().index().equal("1"))
            ),
        )

        # Edge connected to source with
        self.assertEqual(
            sorted([0, 2, 3]),
            sorted(
                medrecord.select_edges(
                    edge().connected_source_with(node().index().equal("0"))
                )
            ),
        )

        # Edge connected with
        self.assertEqual(
            sorted([0, 1]),
            sorted(
                medrecord.select_edges(edge().connected_with(node().index().equal("1")))
            ),
        )

        # Edge has parallel edges with
        self.assertEqual(
            sorted([2, 3]),
            sorted(
                medrecord.select_edges(
                    edge().has_parallel_edges_with(edge().has_attribute("test"))
                )
            ),
        )

        # Edge has parallel edges with self comparison
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().has_parallel_edges_with_self_comparison(
                    edge().attribute("test").equal(edge().attribute("test").sub(1))
                )
            ),
        )

    def test_select_edges_edge_index(self):
        medrecord = create_medrecord()

        # Index greater
        self.assertEqual(
            sorted([2, 3]),
            sorted(medrecord.select_edges(edge().index().greater(1))),
        )

        # Index less
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().index().less(1)),
        )

        # Index greater or equal
        self.assertEqual(
            sorted([1, 2, 3]),
            sorted(medrecord.select_edges(edge().index().greater_or_equal(1))),
        )

        # Index less or equal
        self.assertEqual(
            sorted([0, 1]),
            sorted(medrecord.select_edges(edge().index().less_or_equal(1))),
        )

        # Index equal
        self.assertEqual(
            [1],
            medrecord.select_edges(edge().index().equal(1)),
        )

        # Index not equal
        self.assertEqual(
            sorted([0, 2, 3]),
            sorted(medrecord.select_edges(edge().index().not_equal(1))),
        )

        # Index in
        self.assertEqual(
            [1],
            medrecord.select_edges(edge().index().is_in([1])),
        )

        # Index not in
        self.assertEqual(
            sorted([0, 2, 3]),
            sorted(medrecord.select_edges(edge().index().not_in([1]))),
        )

    def test_select_edges_edges_attribute(self):
        medrecord = create_medrecord()

        # Attribute greater
        self.assertEqual(
            [],
            medrecord.select_edges(edge().attribute("sed").greater("do")),
        )

        # Attribute less
        self.assertEqual(
            [],
            medrecord.select_edges(edge().attribute("sed").less("do")),
        )

        # Attribute greater or equal
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").greater_or_equal("do")),
        )

        # Attribute less or equal
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").less_or_equal("do")),
        )

        # Attribute equal
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").equal("do")),
        )

        # Attribute not equal
        self.assertEqual(
            [],
            medrecord.select_edges(edge().attribute("sed").not_equal("do")),
        )

        # Attribute in
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").is_in(["do"])),
        )

        # Attribute not in
        self.assertEqual(
            [],
            medrecord.select_edges(edge().attribute("sed").not_in(["do"])),
        )

        # Attribute starts with
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").starts_with("d")),
        )

        # Attribute ends with
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").ends_with("o")),
        )

        # Attribute contains
        self.assertEqual(
            [0],
            medrecord.select_edges(edge().attribute("sed").contains("d")),
        )

        # Attribute compare to attribute
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed"))
            ),
        )

        # Attribute compare to attribute add
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed").add("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") == edge().attribute("sed") + "10"
            ),
        )
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed").add("10"))
            ),
        )
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed") != edge().attribute("sed") + "10"
            ),
        )

        # Attribute compare to attribute sub
        # Returns nothing because can't sub a string
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed").sub("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") == edge().attribute("sed") - "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed").sub("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") != edge().attribute("sed") - "10"
            ),
        )

        # Attribute compare to attribute sub
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").sub(10))
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge()
                .attribute("integer")
                .not_equal(edge().attribute("integer").sub(10))
            ),
        )

        # Attribute compare to attribute mul
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed").mul(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") == edge().attribute("sed") * 2
            ),
        )
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed").mul(2))
            ),
        )
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed") != edge().attribute("sed") * 2
            ),
        )

        # Attribute compare to attribute div
        # Returns nothing because can't div a string
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed").div("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") == edge().attribute("sed") / "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed").div("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed") != edge().attribute("sed") / "10"
            ),
        )

        # Attribute compare to attribute div
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").div(2))
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge()
                .attribute("integer")
                .not_equal(edge().attribute("integer").div(2))
            ),
        )

        # Attribute compare to attribute pow
        # Returns nothing because can't pow a string
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem").equal(edge().attribute("lorem").pow("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem") == edge().attribute("lorem") ** "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem").not_equal(edge().attribute("lorem").pow("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem") != edge().attribute("lorem") ** "10"
            ),
        )

        # Attribute compare to attribute pow
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").pow(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge()
                .attribute("integer")
                .not_equal(edge().attribute("integer").pow(2))
            ),
        )

        # Attribute compare to attribute mod
        # Returns nothing because can't mod a string
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem").equal(edge().attribute("lorem").mod("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem") == edge().attribute("lorem") % "10"
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem").not_equal(edge().attribute("lorem").mod("10"))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("lorem") != edge().attribute("lorem") % "10"
            ),
        )

        # Attribute compare to attribute mod
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").mod(2))
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge()
                .attribute("integer")
                .not_equal(edge().attribute("integer").mod(2))
            ),
        )

        # Attribute compare to attribute round
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("sed").round())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").not_equal(edge().attribute("sed").round())
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("float").round())
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("float").not_equal(edge().attribute("float").round())
            ),
        )

        # Attribute compare to attribute ceil
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("float").ceil())
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("float").not_equal(edge().attribute("float").ceil())
            ),
        )

        # Attribute compare to attribute floor
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("float").floor())
            ),
        )
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("float").not_equal(edge().attribute("float").floor())
            ),
        )

        # Attribute compare to attribute abs
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").abs())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("integer").not_equal(edge().attribute("integer").abs())
            ),
        )

        # Attribute compare to attribute sqrt
        self.assertEqual(
            [2],
            medrecord.select_edges(
                edge().attribute("integer").equal(edge().attribute("integer").sqrt())
            ),
        )
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge()
                .attribute("integer")
                .not_equal(edge().attribute("integer").sqrt())
            ),
        )

        # Attribute compare to attribute trim
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("dolor").trim())
            ),
        )

        # Attribute compare to attribute trim_start
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("dolor").trim_start())
            ),
        )

        # Attribute compare to attribute trim_end
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("dolor").trim_end())
            ),
        )

        # Attribute compare to attribute lowercase
        self.assertEqual(
            [0],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("test").lowercase())
            ),
        )

        # Attribute compare to attribute uppercase
        self.assertEqual(
            [],
            medrecord.select_edges(
                edge().attribute("sed").equal(edge().attribute("test").uppercase())
            ),
        )

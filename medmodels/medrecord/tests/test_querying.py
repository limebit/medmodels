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
    def test_select_nodes_node(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("test", ["0"])

        # Node in group
        assert medrecord.select_nodes(node().in_group("test")) == ["0"]

        # Node has attribute
        assert medrecord.select_nodes(node().has_attribute("lorem")) == ["0"]

        # Node has outgoing edge with
        assert medrecord.select_nodes(node().has_outgoing_edge_with(edge().index().equal(0))) == ["0"]

        # Node has incoming edge with
        assert medrecord.select_nodes(node().has_incoming_edge_with(edge().index().equal(0))) == ["1"]

        # Node has edge with
        assert sorted(["0", "1"]) == sorted(medrecord.select_nodes(node().has_edge_with(edge().index().equal(0))))

        # Node has neighbor with
        assert sorted(["0", "1"]) == sorted(medrecord.select_nodes(node().has_neighbor_with(node().index().equal("2"))))
        assert sorted(["0"]) == sorted(medrecord.select_nodes(node().has_neighbor_with(node().index().equal("1"), directed=True)))

        # Node has neighbor with
        assert sorted(["0", "2"]) == sorted(medrecord.select_nodes(node().has_neighbor_with(node().index().equal("1"), directed=False)))

    def test_select_nodes_node_index(self) -> None:
        medrecord = create_medrecord()

        # Index greater
        assert sorted(["2", "3"]) == sorted(medrecord.select_nodes(node().index().greater("1")))

        # Index less
        assert sorted(["0", "1"]) == sorted(medrecord.select_nodes(node().index().less("2")))

        # Index greater or equal
        assert sorted(["1", "2", "3"]) == sorted(medrecord.select_nodes(node().index().greater_or_equal("1")))

        # Index less or equal
        assert sorted(["0", "1", "2"]) == sorted(medrecord.select_nodes(node().index().less_or_equal("2")))

        # Index equal
        assert medrecord.select_nodes(node().index().equal("1")) == ["1"]

        # Index not equal
        assert sorted(["0", "2", "3"]) == sorted(medrecord.select_nodes(node().index().not_equal("1")))

        # Index in
        assert medrecord.select_nodes(node().index().is_in(["1"])) == ["1"]

        # Index not in
        assert sorted(["0", "2", "3"]) == sorted(medrecord.select_nodes(node().index().not_in(["1"])))

        # Index starts with
        assert medrecord.select_nodes(node().index().starts_with("1")) == ["1"]

        # Index ends with
        assert medrecord.select_nodes(node().index().ends_with("1")) == ["1"]

        # Index contains
        assert medrecord.select_nodes(node().index().contains("1")) == ["1"]

    def test_select_nodes_node_attribute(self) -> None:
        medrecord = create_medrecord()

        # Attribute greater
        assert medrecord.select_nodes(node().attribute("lorem").greater("ipsum")) == []
        assert medrecord.select_nodes(node().attribute("lorem") > "ipsum") == []

        # Attribute less
        assert medrecord.select_nodes(node().attribute("lorem").less("ipsum")) == []
        assert medrecord.select_nodes(node().attribute("lorem") < "ipsum") == []

        # Attribute greater or equal
        assert medrecord.select_nodes(node().attribute("lorem").greater_or_equal("ipsum")) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem") >= "ipsum") == ["0"]

        # Attribute less or equal
        assert medrecord.select_nodes(node().attribute("lorem").less_or_equal("ipsum")) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem") <= "ipsum") == ["0"]

        # Attribute equal
        assert medrecord.select_nodes(node().attribute("lorem").equal("ipsum")) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem") == "ipsum") == ["0"]

        # Attribute not equal
        assert medrecord.select_nodes(node().attribute("lorem").not_equal("ipsum")) == []
        assert medrecord.select_nodes(node().attribute("lorem") != "ipsum") == []

        # Attribute in
        assert medrecord.select_nodes(node().attribute("lorem").is_in(["ipsum"])) == ["0"]

        # Attribute not in
        assert medrecord.select_nodes(node().attribute("lorem").not_in(["ipsum"])) == []

        # Attribute starts with
        assert medrecord.select_nodes(node().attribute("lorem").starts_with("ip")) == ["0"]

        # Attribute ends with
        assert medrecord.select_nodes(node().attribute("lorem").ends_with("um")) == ["0"]

        # Attribute contains
        assert medrecord.select_nodes(node().attribute("lorem").contains("su")) == ["0"]

        # Attribute compare to attribute
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem"))) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem"))) == []

        # Attribute compare to attribute add
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").add("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") + "10") == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").add("10"))) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") + "10") == ["0"]

        # Attribute compare to attribute sub
        # Returns nothing because can't sub a string
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").sub("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") + "10") == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").sub("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") - "10") == []

        # Attribute compare to attribute sub
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").sub(10))) == []
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").sub(10))) == ["0"]

        # Attribute compare to attribute mul
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").mul(2))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") * 2) == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").mul(2))) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") * 2) == ["0"]

        # Attribute compare to attribute div
        # Returns nothing because can't div a string
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").div("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") / "10") == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").div("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") / "10") == []

        # Attribute compare to attribute div
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").div(2))) == []
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").div(2))) == ["0"]

        # Attribute compare to attribute pow
        # Returns nothing because can't pow a string
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").pow("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") ** "10") == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").pow("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") ** "10") == []

        # Attribute compare to attribute pow
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").pow(2))) == ["0"]
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").pow(2))) == []

        # Attribute compare to attribute mod
        # Returns nothing because can't mod a string
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").mod("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") == node().attribute("lorem") % "10") == []
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").mod("10"))) == []
        assert medrecord.select_nodes(node().attribute("lorem") != node().attribute("lorem") % "10") == []

        # Attribute compare to attribute mod
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").mod(2))) == ["0"]
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").mod(2))) == []

        # Attribute compare to attribute round
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("lorem").round())) == ["0"]
        assert medrecord.select_nodes(node().attribute("lorem").not_equal(node().attribute("lorem").round())) == []
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("float").round())) == ["0"]
        assert medrecord.select_nodes(node().attribute("float").not_equal(node().attribute("float").round())) == ["0"]

        # Attribute compare to attribute round
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("float").ceil())) == ["0"]
        assert medrecord.select_nodes(node().attribute("float").not_equal(node().attribute("float").ceil())) == ["0"]

        # Attribute compare to attribute floor
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("float").floor())) == []
        assert medrecord.select_nodes(node().attribute("float").not_equal(node().attribute("float").floor())) == ["0"]

        # Attribute compare to attribute abs
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").abs())) == ["0"]
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").abs())) == []

        # Attribute compare to attribute sqrt
        assert medrecord.select_nodes(node().attribute("integer").equal(node().attribute("integer").sqrt())) == ["0"]
        assert medrecord.select_nodes(node().attribute("integer").not_equal(node().attribute("integer").sqrt())) == []

        # Attribute compare to attribute trim
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("dolor").trim())) == ["0"]

        # Attribute compare to attribute trim_start
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("dolor").trim_start())) == []

        # Attribute compare to attribute trim_end
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("dolor").trim_end())) == []

        # Attribute compare to attribute lowercase
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("test").lowercase())) == ["0"]

        # Attribute compare to attribute uppercase
        assert medrecord.select_nodes(node().attribute("lorem").equal(node().attribute("test").uppercase())) == []

    def test_select_edges_edge(self) -> None:
        medrecord = create_medrecord()

        medrecord.add_group("test", edges=[0])

        # Edge connected to target
        assert sorted([1, 2, 3]) == sorted(medrecord.select_edges(edge().connected_target("2")))

        # Edge connected to source
        assert sorted([0, 2, 3]) == sorted(medrecord.select_edges(edge().connected_source("0")))

        # Edge connected
        assert sorted([0, 1]) == sorted(medrecord.select_edges(edge().connected("1")))

        # Edge in group
        assert medrecord.select_edges(edge().in_group("test")) == [0]

        # Edge has attribute
        assert medrecord.select_edges(edge().has_attribute("sed")) == [0]

        # Edge connected to target with
        assert medrecord.select_edges(edge().connected_target_with(node().index().equal("1"))) == [0]

        # Edge connected to source with
        assert sorted([0, 2, 3]) == sorted(medrecord.select_edges(edge().connected_source_with(node().index().equal("0"))))

        # Edge connected with
        assert sorted([0, 1]) == sorted(medrecord.select_edges(edge().connected_with(node().index().equal("1"))))

        # Edge has parallel edges with
        assert sorted([2, 3]) == sorted(medrecord.select_edges(edge().has_parallel_edges_with(edge().has_attribute("test"))))

        # Edge has parallel edges with self comparison
        assert medrecord.select_edges(edge().has_parallel_edges_with_self_comparison(edge().attribute("test").equal(edge().attribute("test").sub(1)))) == [2]

    def test_select_edges_edge_index(self) -> None:
        medrecord = create_medrecord()

        # Index greater
        assert sorted([2, 3]) == sorted(medrecord.select_edges(edge().index().greater(1)))

        # Index less
        assert medrecord.select_edges(edge().index().less(1)) == [0]

        # Index greater or equal
        assert sorted([1, 2, 3]) == sorted(medrecord.select_edges(edge().index().greater_or_equal(1)))

        # Index less or equal
        assert sorted([0, 1]) == sorted(medrecord.select_edges(edge().index().less_or_equal(1)))

        # Index equal
        assert medrecord.select_edges(edge().index().equal(1)) == [1]

        # Index not equal
        assert sorted([0, 2, 3]) == sorted(medrecord.select_edges(edge().index().not_equal(1)))

        # Index in
        assert medrecord.select_edges(edge().index().is_in([1])) == [1]

        # Index not in
        assert sorted([0, 2, 3]) == sorted(medrecord.select_edges(edge().index().not_in([1])))

    def test_select_edges_edges_attribute(self) -> None:
        medrecord = create_medrecord()

        # Attribute greater
        assert medrecord.select_edges(edge().attribute("sed").greater("do")) == []

        # Attribute less
        assert medrecord.select_edges(edge().attribute("sed").less("do")) == []

        # Attribute greater or equal
        assert medrecord.select_edges(edge().attribute("sed").greater_or_equal("do")) == [0]

        # Attribute less or equal
        assert medrecord.select_edges(edge().attribute("sed").less_or_equal("do")) == [0]

        # Attribute equal
        assert medrecord.select_edges(edge().attribute("sed").equal("do")) == [0]

        # Attribute not equal
        assert medrecord.select_edges(edge().attribute("sed").not_equal("do")) == []

        # Attribute in
        assert medrecord.select_edges(edge().attribute("sed").is_in(["do"])) == [0]

        # Attribute not in
        assert medrecord.select_edges(edge().attribute("sed").not_in(["do"])) == []

        # Attribute starts with
        assert medrecord.select_edges(edge().attribute("sed").starts_with("d")) == [0]

        # Attribute ends with
        assert medrecord.select_edges(edge().attribute("sed").ends_with("o")) == [0]

        # Attribute contains
        assert medrecord.select_edges(edge().attribute("sed").contains("d")) == [0]

        # Attribute compare to attribute
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed"))) == [0]
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed"))) == []

        # Attribute compare to attribute add
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed").add("10"))) == []
        assert medrecord.select_edges(edge().attribute("sed") == edge().attribute("sed") + "10") == []
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed").add("10"))) == [0]
        assert medrecord.select_edges(edge().attribute("sed") != edge().attribute("sed") + "10") == [0]

        # Attribute compare to attribute sub
        # Returns nothing because can't sub a string
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed").sub("10"))) == []
        assert medrecord.select_edges(edge().attribute("sed") == edge().attribute("sed") - "10") == []
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed").sub("10"))) == []
        assert medrecord.select_edges(edge().attribute("sed") != edge().attribute("sed") - "10") == []

        # Attribute compare to attribute sub
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").sub(10))) == []
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").sub(10))) == [2]

        # Attribute compare to attribute mul
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed").mul(2))) == []
        assert medrecord.select_edges(edge().attribute("sed") == edge().attribute("sed") * 2) == []
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed").mul(2))) == [0]
        assert medrecord.select_edges(edge().attribute("sed") != edge().attribute("sed") * 2) == [0]

        # Attribute compare to attribute div
        # Returns nothing because can't div a string
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed").div("10"))) == []
        assert medrecord.select_edges(edge().attribute("sed") == edge().attribute("sed") / "10") == []
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed").div("10"))) == []
        assert medrecord.select_edges(edge().attribute("sed") != edge().attribute("sed") / "10") == []

        # Attribute compare to attribute div
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").div(2))) == []
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").div(2))) == [2]

        # Attribute compare to attribute pow
        # Returns nothing because can't pow a string
        assert medrecord.select_edges(edge().attribute("lorem").equal(edge().attribute("lorem").pow("10"))) == []
        assert medrecord.select_edges(edge().attribute("lorem") == edge().attribute("lorem") ** "10") == []
        assert medrecord.select_edges(edge().attribute("lorem").not_equal(edge().attribute("lorem").pow("10"))) == []
        assert medrecord.select_edges(edge().attribute("lorem") != edge().attribute("lorem") ** "10") == []

        # Attribute compare to attribute pow
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").pow(2))) == [2]
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").pow(2))) == []

        # Attribute compare to attribute mod
        # Returns nothing because can't mod a string
        assert medrecord.select_edges(edge().attribute("lorem").equal(edge().attribute("lorem").mod("10"))) == []
        assert medrecord.select_edges(edge().attribute("lorem") == edge().attribute("lorem") % "10") == []
        assert medrecord.select_edges(edge().attribute("lorem").not_equal(edge().attribute("lorem").mod("10"))) == []
        assert medrecord.select_edges(edge().attribute("lorem") != edge().attribute("lorem") % "10") == []

        # Attribute compare to attribute mod
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").mod(2))) == [2]
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").mod(2))) == []

        # Attribute compare to attribute round
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("sed").round())) == [0]
        assert medrecord.select_edges(edge().attribute("sed").not_equal(edge().attribute("sed").round())) == []
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("float").round())) == [2]
        assert medrecord.select_edges(edge().attribute("float").not_equal(edge().attribute("float").round())) == [2]

        # Attribute compare to attribute ceil
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("float").ceil())) == [2]
        assert medrecord.select_edges(edge().attribute("float").not_equal(edge().attribute("float").ceil())) == [2]

        # Attribute compare to attribute floor
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("float").floor())) == []
        assert medrecord.select_edges(edge().attribute("float").not_equal(edge().attribute("float").floor())) == [2]

        # Attribute compare to attribute abs
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").abs())) == [2]
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").abs())) == []

        # Attribute compare to attribute sqrt
        assert medrecord.select_edges(edge().attribute("integer").equal(edge().attribute("integer").sqrt())) == [2]
        assert medrecord.select_edges(edge().attribute("integer").not_equal(edge().attribute("integer").sqrt())) == []

        # Attribute compare to attribute trim
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("dolor").trim())) == [0]

        # Attribute compare to attribute trim_start
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("dolor").trim_start())) == []

        # Attribute compare to attribute trim_end
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("dolor").trim_end())) == []

        # Attribute compare to attribute lowercase
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("test").lowercase())) == [0]

        # Attribute compare to attribute uppercase
        assert medrecord.select_edges(edge().attribute("sed").equal(edge().attribute("test").uppercase())) == []

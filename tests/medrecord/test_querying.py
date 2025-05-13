import unittest
from datetime import datetime, timedelta
from typing import Tuple

from medmodels import MedRecord
from medmodels.medrecord import EdgeIndex, NodeIndex
from medmodels.medrecord.querying import (
    EdgeAttributesTreeOperand,
    EdgeDirection,
    EdgeIndexOperand,
    EdgeIndicesOperand,
    EdgeMultipleAttributesOperand,
    EdgeMultipleValuesOperand,
    EdgeOperand,
    EdgeSingleAttributeOperand,
    EdgeSingleValueOperand,
    NodeAttributesTreeOperand,
    NodeIndexOperand,
    NodeIndicesOperand,
    NodeMultipleAttributesOperand,
    NodeMultipleValuesOperand,
    NodeOperand,
    NodeSingleAttributeOperand,
    NodeSingleValueOperand,
    PyEdgeIndexOperand,
    PyEdgeIndicesOperand,
    PyEdgeMultipleAttributesOperand,
    PyEdgeMultipleValuesOperand,
    PyEdgeSingleAttributeOperand,
    PyEdgeSingleValueOperand,
    PyNodeIndexOperand,
    PyNodeIndicesOperand,
    PyNodeMultipleAttributesOperand,
    PyNodeMultipleValuesOperand,
    PyNodeSingleAttributeOperand,
    PyNodeSingleValueOperand,
    _py_edge_index_comparison_operand_from_edge_index_comparison_operand,
    _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand,
    _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand,
    _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand,
    _py_node_index_comparison_operand_from_node_index_comparison_operand,
    _py_node_indices_comparison_operand_from_node_indices_comparison_operand,
    _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand,
    _py_single_value_comparison_operand_from_single_value_comparison_operand,
)
from medmodels.medrecord.types import MedRecordAttribute, MedRecordValue


def query_node(node: NodeOperand) -> None:
    node.index().equal_to("pat_1")


def query_edge(edge: EdgeOperand) -> None:
    edge.index().equal_to(0)


def query_specific_edge(edge: EdgeOperand, index: EdgeIndex) -> None:
    edge.index().equal_to(index)


class TestPythonTypesConversion(unittest.TestCase):
    def test_node_python_types_conversion(self) -> None:
        medrecord = MedRecord.from_simple_example_dataset()

        cache1: NodeSingleValueOperand

        def query1(node: NodeOperand) -> NodeSingleValueOperand:
            nonlocal cache1
            cache1 = node.attribute("gender").max()
            return cache1

        medrecord.query_nodes(query1)

        type1 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(1)
        )
        type2 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                cache1  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type1, MedRecordValue)
        assert isinstance(type2, PyNodeSingleValueOperand)

        cache2: NodeMultipleValuesOperand

        def query2(node: NodeOperand) -> NodeMultipleValuesOperand:
            nonlocal cache2
            cache2 = node.attribute("age")
            return cache2

        medrecord.query_nodes(query2)

        type3 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            [0, 1]
        )
        type4 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            cache2  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type3, list)
        assert all(isinstance(item, MedRecordValue) for item in type3)
        assert isinstance(type4, PyNodeMultipleValuesOperand)

        cache3: NodeIndicesOperand

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            nonlocal cache3
            cache3 = node.index()
            return cache3

        medrecord.query_nodes(query3)

        type5 = (
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                [0, "node_index"]
            )
        )
        type6 = (
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                cache3  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type5, list)
        assert all(isinstance(item, NodeIndex) for item in type5)
        assert isinstance(type6, PyNodeIndicesOperand)

        cache4: NodeIndexOperand

        def query4(node: NodeOperand) -> NodeIndexOperand:
            nonlocal cache4
            cache4 = node.index().max()
            return cache4

        medrecord.query_nodes(query4)

        type7 = _py_node_index_comparison_operand_from_node_index_comparison_operand(
            "node_index"
        )
        type8 = _py_node_index_comparison_operand_from_node_index_comparison_operand(
            cache4,  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type7, NodeIndex)
        assert isinstance(type8, PyNodeIndexOperand)

        cache5: NodeSingleAttributeOperand

        def query5(node: NodeOperand) -> NodeSingleAttributeOperand:
            nonlocal cache5
            cache5 = node.attributes().max().max()
            return cache5

        medrecord.query_nodes(query5)

        type9 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            "attribute_name",
        )
        type10 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            cache5  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type9, MedRecordAttribute)
        assert isinstance(type10, PyNodeSingleAttributeOperand)

        cache8: NodeMultipleAttributesOperand

        def query8(node: NodeOperand) -> NodeMultipleAttributesOperand:
            nonlocal cache8
            cache8 = node.attributes().max()
            return cache8

        medrecord.query_nodes(query8)

        type15 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            [0, "attribute_name"]
        )
        type16 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            cache8  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type15, list)
        assert all(isinstance(item, MedRecordAttribute) for item in type15)
        assert isinstance(type16, PyNodeMultipleAttributesOperand)

    def test_edge_python_types_conversion(self) -> None:
        medrecord = MedRecord.from_simple_example_dataset()

        cache1: EdgeSingleValueOperand

        def query1(edge: EdgeOperand) -> EdgeSingleValueOperand:
            nonlocal cache1
            cache1 = edge.attribute("time").max()
            return cache1

        medrecord.query_edges(query1)

        type1 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                cache1  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type1, PyEdgeSingleValueOperand)

        cache2: EdgeMultipleValuesOperand

        def query2(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            nonlocal cache2
            cache2 = edge.attribute("age")
            return cache2

        medrecord.query_edges(query2)

        type2 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            cache2  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type2, PyEdgeMultipleValuesOperand)

        cache3: EdgeIndicesOperand

        def query3(edge: EdgeOperand) -> EdgeIndicesOperand:
            nonlocal cache3
            cache3 = edge.index()
            return cache3

        medrecord.query_edges(query3)

        type3 = (
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                [0, 1]
            )
        )
        type4 = (
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                cache3  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type3, list)
        assert all(isinstance(item, EdgeIndex) for item in type3)
        assert isinstance(type4, PyEdgeIndicesOperand)

        cache4: EdgeIndexOperand

        def query4(edge: EdgeOperand) -> EdgeIndexOperand:
            nonlocal cache4
            cache4 = edge.index().max()
            return cache4

        medrecord.query_edges(query4)

        type7 = _py_edge_index_comparison_operand_from_edge_index_comparison_operand(
            cache4  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type7, PyEdgeIndexOperand)

        cache5: EdgeSingleAttributeOperand

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            nonlocal cache5
            cache5 = edge.attributes().max().max()
            return cache5

        medrecord.query_edges(query5)

        type6 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            cache5  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type6, PyEdgeSingleAttributeOperand)

        cache6: EdgeMultipleAttributesOperand

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            nonlocal cache6
            cache6 = edge.attributes().max()
            return cache6

        medrecord.query_edges(query6)
        type7 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            cache6  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )
        assert isinstance(type7, PyEdgeMultipleAttributesOperand)


class TestNodeOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_operand_attribute_simple(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            return node.attribute("gender")

        assert self.medrecord.query_nodes(query) == {"pat_1": "M"}

    def test_node_operand_attributes(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeOperand:
            query_node(node)
            return node.attributes()

        result = {
            key: sorted(value)
            for key, value in self.medrecord.query_nodes(query).items()
        }
        assert result == {"pat_1": ["age", "gender"]}

    def test_node_operand_index(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            return node.index()

        assert self.medrecord.query_nodes(query) == ["pat_1"]

    def test_node_operand_in_group(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query1)) == [
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group(["patient", "diagnosis"])  # Must be in BOTH
            return node.index()

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes_to_group("diagnosis", "pat_1")

        assert sorted(self.medrecord.query_nodes(query2)) == ["pat_1"]

    def test_node_operand_has_attribute(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.has_attribute("age")
            return node.index()

        assert self.medrecord.query_nodes(query1) == ["pat_1"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.has_attribute(["gender", "age"])
            return node.index()

        assert self.medrecord.query_nodes(query2) == ["pat_1"]

    def test_node_operand_edges(self) -> None:
        def query1(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.OUTGOING).index()

        assert 0 in self.medrecord.query_nodes(query1)

        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(("pat_2", "pat_1", {}))

        def query2(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.INCOMING).index()

        assert self.medrecord.query_nodes(query2) == [160]

        def query3(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.BOTH).index()

        assert 0 in self.medrecord.query_nodes(query3)
        assert 160 in self.medrecord.query_nodes(query3)

    def test_node_operand_neighbors(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(("pat_2", "pat_1", {}))

        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.INCOMING)
            return neighbors.index()

        assert self.medrecord.query_nodes(query1) == ["pat_2"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.OUTGOING)
            return neighbors.index()

        assert "procedure_171207006" in self.medrecord.query_nodes(query2)

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.BOTH)
            return neighbors.index()

        assert "pat_2" in self.medrecord.query_nodes(query3)
        assert "procedure_171207006" in self.medrecord.query_nodes(query3)

    def test_node_operand_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.either_or(
                lambda node: node.attribute("age").greater_than(90),
                lambda node: node.attribute("age").less_than(20),
            )
            return node.index()

        assert sorted(self.medrecord.query_nodes(query)) == ["pat_3", "pat_4"]

    def test_node_operand_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.exclude(lambda node: node.attribute("age").greater_than(30))
            return node.index()

        assert sorted(self.medrecord.query_nodes(query)) == ["pat_2", "pat_4"]

    def test_node_operand_clone(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            clone = node.clone()
            node.attribute("age").greater_than(30)
            return clone.index()

        assert sorted(self.medrecord.query_nodes(query)) == [
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]


class TestEdgeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_operand_attribute_simple(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_edge(edge)
            return edge.attribute("time")

        assert self.medrecord.query_edges(query) == {0: datetime(2014, 2, 6, 0, 0)}

    def test_edge_operand_attributes(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_edge(edge)
            return edge.attributes()

        result = {
            key: sorted(value)
            for key, value in self.medrecord.query_edges(query).items()
        }
        assert result == {0: ["duration_days", "time"]}

    def test_edge_operand_index(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

    def test_edge_operand_in_group(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.in_group("patient_diagnosis")
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.in_group(["patient_diagnosis", "treatment"])
            return edge.index()

        self.medrecord.unfreeze_schema()
        self.medrecord.add_group("treatment")
        self.medrecord.add_edges_to_group("treatment", 0)

        assert self.medrecord.query_edges(query2) == [0]

    def test_edge_operand_has_attribute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.has_attribute("time")
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

    def test_edge_operand_source_node(self) -> None:
        def query(edge: EdgeOperand) -> NodeIndicesOperand:
            query_edge(edge)
            return edge.source_node().index()

        result = self.medrecord.query_edges(query)
        assert result == ["pat_1"]

        def query2(edge: EdgeOperand) -> NodeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.source_node().in_group("patient")
            return edge.source_node().index()

        result = self.medrecord.query_edges(query2)
        assert sorted(result) == ["pat_1", "pat_1"]

    def test_edge_operand_target_node(self) -> None:
        def query(edge: EdgeOperand) -> NodeIndicesOperand:
            query_edge(edge)
            return edge.target_node().index()

        result = self.medrecord.query_edges(query)
        assert result == ["diagnosis_82423001"]

        def query2(edge: EdgeOperand) -> NodeIndicesOperand:
            edge.index().is_in([0, 1])
            return edge.target_node().index()

        result = self.medrecord.query_edges(query2)
        assert sorted(result) == ["diagnosis_73595000", "diagnosis_82423001"]

    def test_edge_operand_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.either_or(
                lambda edge: query_edge(edge),
                lambda edge: edge.index().equal_to(1),
            )
            return edge.index()

        assert sorted(self.medrecord.query_edges(query)) == [0, 1]

    def test_edge_operand_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.exclude(lambda edge: edge.attribute("duration_days").greater_than(1))
            return edge.index()

        assert sorted(self.medrecord.query_edges(query)) == [0, 4]

    def test_edge_operand_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            clone = edge.clone()
            edge.attribute("duration_days").less_than(1)
            return clone.index()

        assert sorted(self.medrecord.query_edges(query)) == [0, 1, 2, 3, 4]


class TestNodeMultipleValuesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_multiple_values_operand_numeric(self) -> None:
        assert self.medrecord.query_nodes(lambda node: node.attribute("age").min()) == (
            "pat_4",
            19,
        )

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").max()) == (
            "pat_3",
            96,
        )

        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").mean())
            == 43.2
        )

        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").median())
            == 37
        )

        medrecord_mode = self.medrecord.clone()
        medrecord_mode.unfreeze_schema()
        medrecord_mode.add_nodes(("pat_6", {"age": 22}))
        assert (
            medrecord_mode.query_nodes(lambda node: node.attribute("age").mode()) == 22
        )

        std = float(
            self.medrecord.query_nodes(lambda node: node.attribute("age").std())  # pyright: ignore[reportArgumentType]
        )
        assert round(std, 2) == 27.79

        var = float(
            self.medrecord.query_nodes(lambda node: node.attribute("age").var())  # pyright: ignore[reportArgumentType]
        )
        assert round(var, 2) == 772.56

        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").count()) == 5
        )

        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").sum()) == 216
        )

        def query_random(node: NodeOperand) -> NodeSingleValueOperand:
            query_node(node)
            return node.attribute("age").random()

        assert self.medrecord.query_nodes(query_random) == ("pat_1", 42)

    def test_node_multiple_values_operand_datatypes(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("gender").is_string()
            return node.index()

        assert self.medrecord.query_nodes(query1) == ["pat_1"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").is_string()
            return node.index()

        assert self.medrecord.query_nodes(query2) == []

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").is_int()
            return node.index()

        assert self.medrecord.query_nodes(query3) == ["pat_1"]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"datetime_attribute": datetime(2023, 1, 1)})
        )

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.index().equal_to("pat_6")
            node.attribute("datetime_attribute").is_datetime()
            return node.index()

        assert self.medrecord.query_nodes(query4) == ["pat_6"]

        self.medrecord.add_nodes(("pat_7", {"float_attribute": 2.3}))

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            # node.index().equal_to("pat_7")
            node.attribute("float_attribute").is_float()
            return node.index()

        assert self.medrecord.query_nodes(query5) == ["pat_7"]

        self.medrecord.add_nodes(("pat_8", {"duration_attribute": timedelta(days=3)}))

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("duration_attribute").is_duration()
            return node.index()

        # TODO(@JakobKrauskopf #266): revisit after implementing duration in datasets
        assert self.medrecord.query_nodes(query6) == ["pat_8"]

        self.medrecord.add_nodes(("pat_9", {"bool_attribute": True}))

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("bool_attribute").is_bool()
            return node.index()

        assert self.medrecord.query_nodes(query7) == ["pat_9"]

        self.medrecord.add_nodes(("pat_10", {"null_attribute": None}))

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("null_attribute").is_null()
            return node.index()

        assert self.medrecord.query_nodes(query8) == ["pat_10"]

    def test_node_multiple_values_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_max()
            return node.index()

        assert self.medrecord.query_nodes(query1) == ["pat_3"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_min()
            return node.index()

        assert self.medrecord.query_nodes(query2) == ["pat_4"]

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").greater_than(90)
            return node.index()

        assert self.medrecord.query_nodes(query3) == ["pat_3"]

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").less_than(20)
            return node.index()

        assert self.medrecord.query_nodes(query4) == ["pat_4"]

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").equal_to(42)
            return node.index()

        assert self.medrecord.query_nodes(query5) == ["pat_1"]

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").not_equal_to(42)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query6)) == [
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_in([42, 19])
            return node.index()

        assert sorted(self.medrecord.query_nodes(query7)) == ["pat_1", "pat_4"]

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_not_in([42, 19])
            return node.index()

        assert sorted(self.medrecord.query_nodes(query8)) == ["pat_2", "pat_3", "pat_5"]

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").less_than_or_equal_to(42)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query9)) == [
            "pat_1",
            "pat_2",
            "pat_4",
            "pat_5",
        ]

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").greater_than_or_equal_to(42)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query10)) == ["pat_1", "pat_3"]

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").starts_with("1")
            return node.index()

        assert self.medrecord.query_nodes(query11) == ["pat_4"]

        def query12(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").ends_with("9")
            return node.index()

        assert self.medrecord.query_nodes(query12) == ["pat_4"]

        def query13(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("gender").contains("M")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query13)) == [
            "pat_1",
            "pat_4",
            "pat_5",
        ]

    def test_node_multiple_values_operand_operations(self) -> None:
        def query1(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.add(10)
            return age

        assert self.medrecord.query_nodes(query1) == {"pat_1": 52}

        def query2(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(10)
            return age

        assert self.medrecord.query_nodes(query2) == {"pat_1": 32}

        def query3(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.multiply(10)
            return age

        assert self.medrecord.query_nodes(query3) == {"pat_1": 420}

        def query4(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(10)
            return age

        assert self.medrecord.query_nodes(query4) == {"pat_1": 4.2}

        def query5(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.modulo(10)
            return age

        assert self.medrecord.query_nodes(query5) == {"pat_1": 2}

        def query6(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.power(2)
            return age

        assert self.medrecord.query_nodes(query6) == {"pat_1": 1764}

        def query7(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.floor()
            return age

        assert self.medrecord.query_nodes(query7) == {"pat_1": 8}

        def query8(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.ceil()
            return age

        assert self.medrecord.query_nodes(query8) == {"pat_1": 9}

        def query9(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.round()
            return age

        assert self.medrecord.query_nodes(query9) == {"pat_1": 8}

        def query10(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.sqrt()
            return age

        result = self.medrecord.query_nodes(query10)
        result = {key: round(float(value), 2) for key, value in result.items()}  # pyright: ignore[reportArgumentType]
        assert result == {"pat_1": 6.48}

        def query11(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(45)
            age.absolute()
            return age

        assert self.medrecord.query_nodes(query11) == {"pat_1": 3}

        def query12(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            age = node.attribute("gender")
            age.lowercase()
            return age

        assert self.medrecord.query_nodes(query12) == {"pat_1": "m"}

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"spacing": " hello "}))

        def query13(node: NodeOperand) -> NodeMultipleValuesOperand:
            age = node.attribute("spacing")
            age.uppercase()
            return age

        assert self.medrecord.query_nodes(query13) == {"pat_6": " HELLO "}

        def query14(node: NodeOperand) -> NodeMultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim()
            return age

        assert self.medrecord.query_nodes(query14) == {"pat_6": "hello"}

        def query15(node: NodeOperand) -> NodeMultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim_start()
            return age

        assert self.medrecord.query_nodes(query15) == {"pat_6": "hello "}

        def query16(node: NodeOperand) -> NodeMultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim_end()
            return age

        assert self.medrecord.query_nodes(query16) == {"pat_6": " hello"}

        def query17(node: NodeOperand) -> NodeMultipleValuesOperand:
            age = node.attribute("spacing")
            age.slice(0, 3)
            return age

        assert self.medrecord.query_nodes(query17) == {"pat_6": " he"}

        def query18(node: NodeOperand) -> NodeMultipleValuesOperand:
            node.attribute("age").either_or(
                lambda attribute: attribute.greater_than(90),
                lambda attribute: attribute.less_than(20),
            )
            return node.attribute("age")

        result = self.medrecord.query_nodes(query18)
        assert self.medrecord.query_nodes(query18) == {"pat_3": 96, "pat_4": 19}

        def query19(node: NodeOperand) -> NodeMultipleValuesOperand:
            node.attribute("age").exclude(
                lambda attribute: attribute.less_than(90),
            )
            return node.attribute("age")

        assert self.medrecord.query_nodes(query19) == {"pat_3": 96}

        def query20(node: NodeOperand) -> NodeMultipleValuesOperand:
            query_node(node)
            clone = node.attribute("age").clone()
            node.attribute("age").add(10)
            return clone

        assert self.medrecord.query_nodes(query20) == {"pat_1": 42}


class TestEdgeMultipleValuesOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            [
                (
                    "pat_1",
                    "pat_2",
                    {
                        "string_attribute": " Hello ",
                        "float_attribute": 50.5,
                        "integer_attribute": 5,
                    },
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"bool_attribute": True},
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"null_attribute": None},
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"duration_attribute": timedelta(hours=2)},
                ),
            ]
        )

    def test_edge_multiple_values_operand_numeric(self) -> None:
        def query_min(edge: EdgeOperand) -> EdgeSingleValueOperand:
            query_specific_edge(edge, 4)
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.min()

        assert self.medrecord.query_edges(query_min) == (4, 0)

        def query_max(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.max()

        assert self.medrecord.query_edges(query_max) == (47, 3416)

        def query_mean(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.mean()

        mean_val = float(self.medrecord.query_edges(query_mean))  # pyright: ignore[reportArgumentType]
        assert round(mean_val, 2) == 405.02

        def query_median(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.median()

        assert self.medrecord.query_edges(query_median) == 315

        def query_mode(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.mode()

        assert self.medrecord.query_edges(query_mode) == 371

        def query_std(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.std()

        std = float(self.medrecord.query_edges(query_std))  # pyright: ignore[reportArgumentType]
        assert round(std, 2) == 605.22

        def query_var(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.var()

        var = float(self.medrecord.query_edges(query_var))  # pyright: ignore[reportArgumentType]
        assert round(var, 2) == 366285.42

        def query_count(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.count()

        assert self.medrecord.query_edges(query_count) == 43

        def query_sum(edge: EdgeOperand) -> EdgeSingleValueOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.sum()

        assert self.medrecord.query_edges(query_sum) == 17416

        def query_random(edge: EdgeOperand) -> EdgeSingleValueOperand:
            query_specific_edge(edge, 3)
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.random()

        assert self.medrecord.query_edges(query_random) == (3, 12)

    def test_edge_multiple_values_operand_datatypes(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("string_attribute").is_string()
            return edge.index()

        assert self.medrecord.query_edges(query1) == [160]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("duration_days").is_string()
            return edge.index()

        assert self.medrecord.query_edges(query2) == []

        def query3(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("integer_attribute").is_int()
            return edge.index()

        assert self.medrecord.query_edges(query3) == [160]

        def query4(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.attribute("time").is_datetime()
            return edge.index()

        assert self.medrecord.query_edges(query4) == [0]

        def query5(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("float_attribute").is_float()
            return edge.index()

        assert self.medrecord.query_edges(query5) == [160]

        def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("duration_attribute").is_duration()
            return edge.index()

        assert self.medrecord.query_edges(query6) == [163]

        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("bool_attribute").is_bool()
            return edge.index()

        assert self.medrecord.query_edges(query7) == [161]

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("null_attribute").is_null()
            return edge.index()

        assert self.medrecord.query_edges(query8) == [162]

    def test_edge_multiple_values_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndicesOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            attribute.is_max()
            return edge.index()

        assert self.medrecord.query_edges(query1) == [47]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(10)
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            attribute.is_min()
            return edge.index()

        assert sorted(self.medrecord.query_edges(query2)) == [4, 6]

        def query3(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").greater_than(12)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query3)) == [1, 2]

        def query4(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_specific_edge(edge, 4)
            edge.attribute("duration_days").less_than(12)
            return edge.index()

        assert self.medrecord.query_edges(query4) == [4]

        def query5(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(10)
            edge.attribute("duration_days").equal_to(0)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query5)) == [4, 6]

        def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(2)
            edge.attribute("duration_days").not_equal_to(0)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query6)) == [0, 1]

        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").is_in([0, 12])
            return edge.index()

        assert sorted(self.medrecord.query_edges(query7)) == [3, 4]

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").is_not_in([0, 12])
            return edge.index()

        assert sorted(self.medrecord.query_edges(query8)) == [0, 1, 2]

        def query9(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").less_than_or_equal_to(12)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query9)) == [3, 4]

        def query10(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").greater_than_or_equal_to(12)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query10)) == [1, 2, 3]

        def query11(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("string_attribute").starts_with(" He")
            return edge.index()

        assert self.medrecord.query_edges(query11) == [160]

        def query12(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("string_attribute").ends_with("lo ")
            return edge.index()

        assert self.medrecord.query_edges(query12) == [160]

        def query13(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("string_attribute").contains("Hello")
            return edge.index()

        assert self.medrecord.query_edges(query13) == [160]

    def test_edge_multiple_values_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.add(10)
            return duration

        assert self.medrecord.query_edges(query1) == {3: 22}

        def query2(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.subtract(1)
            return duration

        assert self.medrecord.query_edges(query2) == {3: 11}

        def query3(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.multiply(5)
            return duration

        assert self.medrecord.query_edges(query3) == {3: 60}

        def query4(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            return duration

        assert self.medrecord.query_edges(query4) == {3: 3}

        def query5(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.modulo(3)
            return duration

        assert self.medrecord.query_edges(query5) == {3: 0}

        def query6(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.power(3)
            return duration

        assert self.medrecord.query_edges(query6) == {3: 1728}

        def query7(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(5)
            duration.floor()
            return duration

        assert self.medrecord.query_edges(query7) == {3: 2}

        def query8(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            duration.ceil()
            return duration

        assert self.medrecord.query_edges(query8) == {3: 3}

        def query9(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            duration.round()
            return duration

        assert self.medrecord.query_edges(query9) == {3: 3}

        def query10(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.power(0.5)
            return duration

        result = self.medrecord.query_edges(query10)
        result = {key: round(float(value), 2) for key, value in result.items()}  # pyright: ignore[reportArgumentType]
        assert result == {3: 3.46}

        def query11(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.subtract(15)
            duration.absolute()
            return duration

        assert self.medrecord.query_edges(query11) == {3: 3}

        def query12(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.lowercase()
            return label

        assert self.medrecord.query_edges(query12) == {160: " hello "}

        def query13(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.uppercase()
            return label

        assert self.medrecord.query_edges(query13) == {160: " HELLO "}

        def query14(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.trim()
            return label

        assert self.medrecord.query_edges(query14) == {160: "Hello"}

        def query15(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.trim_start()
            return label

        assert self.medrecord.query_edges(query15) == {160: "Hello "}

        def query16(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.trim_end()
            return label

        assert self.medrecord.query_edges(query16) == {160: " Hello"}

        def query17(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            label = edge.attribute("string_attribute")
            label.slice(1, 3)
            return label

        assert self.medrecord.query_edges(query17) == {160: "He"}

        def query17_time_add(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 0)
            time_attr = edge.attribute("time")
            time_attr.add(timedelta(days=5))
            return time_attr

        assert self.medrecord.query_edges(query17_time_add) == {
            0: datetime(2014, 2, 11)
        }

        def query18_time_subtract(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 0)
            time_attr = edge.attribute("time")
            time_attr.subtract(timedelta(hours=6))
            return time_attr

        assert self.medrecord.query_edges(query18_time_subtract) == {
            0: datetime(2014, 2, 5, 18, 0)
        }

        def query19(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").either_or(
                lambda attribute: attribute.equal_to(0),
                lambda attribute: attribute.equal_to(12),
            )
            return edge.index()

        assert sorted(self.medrecord.query_edges(query19)) == [3, 4]

        def query20(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.attribute("duration_days").exclude(
                lambda attribute: attribute.less_than(10),
            )
            return edge.index()

        assert sorted(self.medrecord.query_edges(query20)) == [0, 1, 2, 3]

        def query21(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            clone = edge.attribute("duration_days").clone()
            edge.attribute("duration_days").add(10)
            return clone

        assert self.medrecord.query_edges(query21) == {3: 12}

        def query22(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            query_specific_edge(edge, 3)
            attribute = edge.attribute("duration_days")
            attribute.sqrt()
            return attribute

        result = self.medrecord.query_edges(query22)
        result = {key: round(float(value), 2) for key, value in result.items()}  # pyright: ignore[reportArgumentType]
        assert result == {3: 3.46}


class TestNodeSingleValueOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_single_value_operand_datatypes(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("gender").max().is_string()
            return node.index()

        assert self.medrecord.query_nodes(query1) == ["pat_1"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").max().is_string()
            return node.index()

        assert self.medrecord.query_nodes(query2) == []

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").max().is_int()
            return node.index()

        assert self.medrecord.query_nodes(query3) == ["pat_1"]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"datetime_attribute": datetime(2023, 1, 1)})
        )

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("datetime_attribute").max().is_datetime()
            return node.index()

        assert self.medrecord.query_nodes(query4) == ["pat_6"]

        self.medrecord.add_nodes(("pat_7", {"float_attribute": 2.3}))

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("float_attribute").max().is_float()
            return node.index()

        assert self.medrecord.query_nodes(query5) == ["pat_7"]

        self.medrecord.add_nodes(("pat_8", {"duration_attribute": timedelta(days=3)}))

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("duration_attribute").max().is_duration()
            return node.index()

        # TODO(@JakobKrauskopf #266): revisit after implementing duration in datasets
        assert self.medrecord.query_nodes(query6) == ["pat_8"]

        self.medrecord.add_nodes(("pat_9", {"bool_attribute": True}))

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("bool_attribute").max().is_bool()
            return node.index()

        assert self.medrecord.query_nodes(query7) == ["pat_9"]

        self.medrecord.add_nodes(("pat_10", {"null_attribute": None}))

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("null_attribute").max().is_null()
            return node.index()

        assert self.medrecord.query_nodes(query8) == ["pat_10"]

    def test_node_single_value_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.greater_than(90)
            return maximum

        assert self.medrecord.query_nodes(query1) == ("pat_3", 96)

        def query2(node: NodeOperand) -> NodeSingleValueOperand:
            minimum = node.attribute("age").min()
            minimum.less_than(20)
            return minimum

        assert self.medrecord.query_nodes(query2) == ("pat_4", 19)

        def query3(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.equal_to(96)
            return maximum

        assert self.medrecord.query_nodes(query3) == ("pat_3", 96)

        def query4(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.not_equal_to(42)
            return maximum

        assert self.medrecord.query_nodes(query4) == ("pat_3", 96)

        def query5(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.is_in([96, 19])
            return maximum

        assert self.medrecord.query_nodes(query5) == ("pat_3", 96)

        def query6(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.is_not_in([42, 19])
            return maximum

        assert self.medrecord.query_nodes(query6) == ("pat_3", 96)

        def query7(node: NodeOperand) -> NodeSingleValueOperand:
            minimum = node.attribute("age").min()
            minimum.less_than_or_equal_to(42)
            return minimum

        assert self.medrecord.query_nodes(query7) == ("pat_4", 19)

        def query8(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.greater_than_or_equal_to(96)
            return maximum

        assert self.medrecord.query_nodes(query8) == ("pat_3", 96)

        def query9(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.starts_with(9)
            return maximum

        assert self.medrecord.query_nodes(query9) == ("pat_3", 96)

        def query10(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.ends_with(6)
            return maximum

        assert self.medrecord.query_nodes(query10) == ("pat_3", 96)

        def query11(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.contains(9)
            return maximum

        assert self.medrecord.query_nodes(query11) == ("pat_3", 96)

    def test_node_single_value_operand_operations(self) -> None:
        def query1(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.add(10)
            return maximum

        assert self.medrecord.query_nodes(query1) == ("pat_3", 106)

        def query2(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.subtract(10)
            return maximum

        assert self.medrecord.query_nodes(query2) == ("pat_3", 86)

        def query3(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.multiply(10)
            return maximum

        assert self.medrecord.query_nodes(query3) == ("pat_3", 960)

        def query4(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.divide(10)
            return maximum

        assert self.medrecord.query_nodes(query4) == ("pat_3", 9.6)

        def query5(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.modulo(10)
            return maximum

        assert self.medrecord.query_nodes(query5) == ("pat_3", 6)

        def query6(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.power(2)
            return maximum

        assert self.medrecord.query_nodes(query6) == ("pat_3", 9216)

        def query7(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.floor()
            return maximum

        assert self.medrecord.query_nodes(query7) == ("pat_3", 19)

        def query8(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.ceil()
            return maximum

        assert self.medrecord.query_nodes(query8) == ("pat_3", 20)

        def query9(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.round()
            return maximum

        assert self.medrecord.query_nodes(query9) == ("pat_3", 19)

        def query10(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.sqrt()
            return maximum

        result: Tuple[NodeIndex, float] = self.medrecord.query_nodes(query10)  # pyright: ignore[reportAssignmentType]
        result = (result[0], round(float(result[1]), 2))

        assert result == ("pat_3", 9.8)

        def query11(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.subtract(100)
            maximum.absolute()
            return maximum

        assert self.medrecord.query_nodes(query11) == ("pat_3", 4)

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"spacing": " Hello "}))

        def query12(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.lowercase()
            return maximum

        assert self.medrecord.query_nodes(query12) == ("pat_6", " hello ")

        def query13(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.uppercase()
            return maximum

        assert self.medrecord.query_nodes(query13) == ("pat_6", " HELLO ")

        def query14(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim()
            return maximum

        assert self.medrecord.query_nodes(query14) == ("pat_6", "Hello")

        def query15(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim_start()
            return maximum

        assert self.medrecord.query_nodes(query15) == ("pat_6", "Hello ")

        def query16(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim_end()
            return maximum

        assert self.medrecord.query_nodes(query16) == ("pat_6", " Hello")

        def query17(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("spacing").max()
            maximum.slice(0, 3)
            return maximum

        assert self.medrecord.query_nodes(query17) == ("pat_6", " He")

        def query18(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.either_or(
                lambda value: value.greater_than(90),
                lambda value: value.less_than(20),
            )
            return maximum

        assert self.medrecord.query_nodes(query18) == ("pat_3", 96)

        def query19(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            maximum.exclude(
                lambda value: value.less_than(90),
            )
            return maximum

        assert self.medrecord.query_nodes(query19) == ("pat_3", 96)

        def query20(node: NodeOperand) -> NodeSingleValueOperand:
            maximum = node.attribute("age").max()
            clone = maximum.clone()
            maximum.add(10)
            return clone

        assert self.medrecord.query_nodes(query20) == ("pat_3", 96)


class TestEdgeSingleValueOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            [
                (
                    "pat_1",
                    "pat_2",
                    {
                        "string_attribute": " Hello ",
                        "float_attribute": 50.5,
                        "integer_attribute": 5,
                    },
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"bool_attribute": True},
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"null_attribute": None},
                ),
                (
                    "pat_1",
                    "pat_2",
                    {"duration_attribute": timedelta(hours=2)},
                ),
            ]
        )

    def test_edge_single_value_operand_datatypes(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.is_string()
            return value

        assert self.medrecord.query_edges(query1) == (160, " Hello ")

        def query2(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.is_int()
            return value

        assert self.medrecord.query_edges(query2) == (160, 5)

        def query3(edge: EdgeOperand) -> EdgeSingleValueOperand:
            query_edge(edge)
            value = edge.attribute("time").max()
            value.is_datetime()
            return value

        assert self.medrecord.query_edges(query3) == (0, datetime(2014, 2, 6, 0, 0))

        def query4(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("float_attribute").max()
            value.is_float()
            return value

        assert self.medrecord.query_edges(query4) == (160, 50.5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("duration_attribute").max()
            value.is_duration()
            return value

        assert self.medrecord.query_edges(query5) == (163, timedelta(hours=2))

        def query6(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("bool_attribute").max()
            value.is_bool()
            return value

        assert self.medrecord.query_edges(query6) == (161, True)

        def query7(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("null_attribute").min()
            value.is_null()
            return value

        assert self.medrecord.query_edges(query7) == (162, None)

        def query8(edge: EdgeOperand) -> EdgeSingleValueOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").max()
            value.is_float()
            return value

        assert self.medrecord.query_edges(query8) == (47, 3416)

    def test_edge_single_value_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.greater_than(4)
            return value

        assert self.medrecord.query_edges(query1) == (160, 5)

        def query2(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.less_than(10)
            return value

        assert self.medrecord.query_edges(query2) == (160, 5)

        def query3(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.equal_to(5)
            return value

        assert self.medrecord.query_edges(query3) == (160, 5)

        def query4(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.not_equal_to(10)
            return value

        assert self.medrecord.query_edges(query4) == (160, 5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.is_in([1, 3, 5, 7])
            return value

        assert self.medrecord.query_edges(query5) == (160, 5)

        def query6(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()  # 5
            value.is_not_in([1, 3, 7, 9])
            return value

        assert self.medrecord.query_edges(query6) == (160, 5)

        def query7(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.less_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query7) == (160, 5)

        def query8(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.greater_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query8) == (160, 5)

        def query9(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.starts_with(" H")
            return value

        assert self.medrecord.query_edges(query9) == (160, " Hello ")

        def query10(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.ends_with("o ")
            return value

        assert self.medrecord.query_edges(query10) == (160, " Hello ")

        def query11(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.contains("ell")
            return value

        assert self.medrecord.query_edges(query11) == (160, " Hello ")

    def test_edge_single_value_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.add(10)
            return value

        assert self.medrecord.query_edges(query1) == (160, 15)

        def query2(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.subtract(2)
            return value

        assert self.medrecord.query_edges(query2) == (160, 3)

        def query3(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.multiply(3)
            return value

        assert self.medrecord.query_edges(query3) == (160, 15)

        def query4(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.divide(2)
            return value

        assert self.medrecord.query_edges(query4) == (160, 2.5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.modulo(3)
            return value

        assert self.medrecord.query_edges(query5) == (160, 2)

        def query6(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.power(2)
            return value

        assert self.medrecord.query_edges(query6) == (160, 25)

        def query7(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("float_attribute").max()
            value.floor()
            return value

        assert self.medrecord.query_edges(query7) == (160, 50.0)

        def query8(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("float_attribute").max()
            value.ceil()
            return value

        assert self.medrecord.query_edges(query8) == (160, 51.0)

        def query9(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("float_attribute").max()
            value.round()
            return value

        assert self.medrecord.query_edges(query9) == (160, 51.0)

        def query10(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.power(0.5)
            return value

        result: Tuple[EdgeIndex, float] = self.medrecord.query_edges(query10)  # pyright: ignore[reportAssignmentType]
        result = (result[0], round(float(result[1]), 2))
        assert result[0] == 160
        assert result[1] == 2.24

        def query11(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.subtract(10)
            value.absolute()
            return value

        assert self.medrecord.query_edges(query11) == (160, 5)

        def query12(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.lowercase()
            return value

        assert self.medrecord.query_edges(query12) == (160, " hello ")

        def query13(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.uppercase()
            return value

        assert self.medrecord.query_edges(query13) == (160, " HELLO ")

        def query14(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.trim()
            return value

        assert self.medrecord.query_edges(query14) == (160, "Hello")

        def query15(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.trim_start()
            return value

        assert self.medrecord.query_edges(query15) == (160, "Hello ")

        def query16(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()
            value.trim_end()
            return value

        assert self.medrecord.query_edges(query16) == (160, " Hello")

        def query17(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("string_attribute").max()  # " Hello "
            value.slice(1, 4)  # Slice "Hel"
            return value

        assert self.medrecord.query_edges(query17) == (160, "Hel")

        def query18(edge: EdgeOperand) -> EdgeSingleValueOperand:
            query_specific_edge(edge, 0)
            value = edge.attribute("time").max()
            value.add(timedelta(days=5))
            return value

        assert self.medrecord.query_edges(query18) == (0, datetime(2014, 2, 11))

        def query19(edge: EdgeOperand) -> EdgeSingleValueOperand:
            query_specific_edge(edge, 0)
            value = edge.attribute("time").max()
            value.subtract(timedelta(hours=1))
            return value

        assert self.medrecord.query_edges(query19) == (0, datetime(2014, 2, 5, 23, 0))

        def query20(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.either_or(
                lambda value: value.greater_than(3),
                lambda value: value.less_than(2),
            )
            return value

        assert self.medrecord.query_edges(query20) == (160, 5)

        def query21(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.exclude(
                lambda value: value.less_than(3),
            )
            return value

        assert self.medrecord.query_edges(query21) == (160, 5)

        def query22(edge: EdgeOperand) -> EdgeSingleValueOperand:
            original_val = edge.attribute("integer_attribute").max()
            cloned_val = original_val.clone()
            original_val.add(50)
            return cloned_val

        assert self.medrecord.query_edges(query22) == (160, 5)

        def query23(edge: EdgeOperand) -> EdgeSingleValueOperand:
            value = edge.attribute("integer_attribute").max()
            value.sqrt()
            return value

        result: Tuple[EdgeIndex, float] = self.medrecord.query_edges(query23)  # pyright: ignore[reportAssignmentType]
        result = (result[0], round(float(result[1]), 2))
        assert result[0] == 160


class TestNodeIndicesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_indices_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.index().equal_to("pat_1")
            return node.index()

        assert self.medrecord.query_nodes(query1) == ["pat_1"]

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.index().not_equal_to("pat_1")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query2)) == [
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {}))
        self.medrecord.add_nodes((1, {}))

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.index().is_in([0, 1])
            return node.index()

        assert sorted(self.medrecord.query_nodes(query3)) == [0, 1]

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.index().is_not_in(["pat_1", "pat_2"])
            return node.index()

        assert sorted(self.medrecord.query_nodes(query4)) == ["pat_3", "pat_4", "pat_5"]

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.index().less_than(1)
            return node.index()

        assert self.medrecord.query_nodes(query5) == [0]

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.index().less_than_or_equal_to(1)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query6)) == [0, 1]

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.index().greater_than(0)
            return node.index()

        assert self.medrecord.query_nodes(query7) == [1]

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.index().greater_than_or_equal_to(0)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query8)) == [0, 1]

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            node.index().starts_with("pat_")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query9)) == [
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            node.index().ends_with("_1")
            return node.index()

        assert self.medrecord.query_nodes(query10) == ["pat_1"]

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            node.index().contains("at")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query11)) == [
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

        def query12(node: NodeOperand) -> NodeIndicesOperand:
            node.index().is_string()
            return node.index()

        assert 0 not in self.medrecord.query_nodes(query12)
        assert "pat_1" in self.medrecord.query_nodes(query12)

        def query13(node: NodeOperand) -> NodeIndicesOperand:
            node.index().is_int()
            return node.index()

        assert sorted(self.medrecord.query_nodes(query13)) == [0, 1]

        def query14(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_in([0, 1])
            indices.is_max()
            return indices

        assert self.medrecord.query_nodes(query14) == [1]

        def query15(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_in([0, 1])
            indices.is_min()
            return indices

        assert self.medrecord.query_nodes(query15) == [0]

    def test_node_indices_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((10, {}))
        self.medrecord.add_nodes((11, {}))

        def query1(node: NodeOperand) -> NodeIndexOperand:
            indices = node.index()
            indices.is_int()
            return indices.max()

        assert self.medrecord.query_nodes(query1) == 11

        def query2(node: NodeOperand) -> NodeIndexOperand:
            indices = node.index()
            indices.is_int()
            return indices.min()

        assert self.medrecord.query_nodes(query2) == 10

        def query3(node: NodeOperand) -> NodeIndexOperand:
            return node.index().count()

        assert self.medrecord.query_nodes(query3) == 75

        def query4(node: NodeOperand) -> NodeIndexOperand:
            indices = node.index()
            indices.is_int()
            return indices.sum()

        assert self.medrecord.query_nodes(query4) == 21

        def query5(node: NodeOperand) -> NodeIndexOperand:
            indices = node.index()
            indices.greater_than(10)
            return indices.random()

        assert self.medrecord.query_nodes(query5) == 11

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.add(2)
            return indices

        assert sorted(self.medrecord.query_nodes(query6)) == [12, 13]

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.subtract(2)
            return indices

        assert sorted(self.medrecord.query_nodes(query7)) == [8, 9]

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.multiply(2)
            return indices

        assert sorted(self.medrecord.query_nodes(query8)) == [20, 22]

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.modulo(2)
            return indices

        assert sorted(self.medrecord.query_nodes(query9)) == [0, 1]

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.power(2)
            return indices

        assert sorted(self.medrecord.query_nodes(query10)) == [100, 121]

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.is_int()
            indices.subtract(12)
            indices.absolute()
            return indices

        assert sorted(self.medrecord.query_nodes(query11)) == [1, 2]

        self.medrecord.add_nodes((" Hello ", {}))

        def query12(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.trim()
            return indices

        assert self.medrecord.query_nodes(query12) == ["Hello"]

        def query13(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.trim_start()
            return indices

        assert self.medrecord.query_nodes(query13) == ["Hello "]

        def query14(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.trim_end()
            return indices

        assert self.medrecord.query_nodes(query14) == [" Hello"]

        def query15(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.slice(0, 3)
            return indices

        assert self.medrecord.query_nodes(query15) == [" He"]

        def query16(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.lowercase()
            return indices

        assert self.medrecord.query_nodes(query16) == [" hello "]

        def query17(node: NodeOperand) -> NodeIndicesOperand:
            indices = node.index()
            indices.contains(" ")
            indices.uppercase()
            return indices

        assert self.medrecord.query_nodes(query17) == [" HELLO "]

        def query18(node: NodeOperand) -> NodeIndicesOperand:
            node.index().either_or(
                lambda index: index.equal_to("pat_1"),
                lambda index: index.equal_to("pat_2"),
            )
            return node.index()

        assert sorted(self.medrecord.query_nodes(query18)) == ["pat_1", "pat_2"]

        def query19(node: NodeOperand) -> NodeIndicesOperand:
            node.index().exclude(
                lambda index: index.is_string(),
            )
            return node.index()

        assert sorted(self.medrecord.query_nodes(query19)) == [10, 11]

        def query20(node: NodeOperand) -> NodeIndicesOperand:
            node.index().is_int()
            clone = node.index().clone()
            node.index().add(10)
            return clone

        assert sorted(self.medrecord.query_nodes(query20)) == [10, 11]


class TestNodeIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_index_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.equal_to("pat_5")
            return maximum

        assert self.medrecord.query_nodes(query1) == "pat_5"

        def query2(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.not_equal_to("pat_1")
            return maximum

        assert self.medrecord.query_nodes(query2) == "pat_5"

        def query3(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.is_in(["pat_1", "pat_5"])
            return maximum

        assert self.medrecord.query_nodes(query3) == "pat_5"

        def query4(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.is_not_in(["pat_1", "pat_2"])
            return maximum

        assert self.medrecord.query_nodes(query4) == "pat_5"

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {}))
        self.medrecord.add_nodes((1, {}))

        def query5(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            minimum = node.index().min()
            minimum.less_than(1)
            return minimum

        assert self.medrecord.query_nodes(query5) == 0

        def query6(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.less_than_or_equal_to(1)
            return maximum

        assert self.medrecord.query_nodes(query6) == 1

        def query7(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.greater_than(0)
            return maximum

        assert self.medrecord.query_nodes(query7) == 1

        def query8(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.greater_than_or_equal_to(0)
            return maximum

        assert self.medrecord.query_nodes(query8) == 1

        def query9(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.starts_with("pat_")
            return maximum

        assert self.medrecord.query_nodes(query9) == "pat_5"

        def query10(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            minimum = node.index().min()
            minimum.ends_with("_1")
            return minimum

        assert self.medrecord.query_nodes(query10) == "pat_1"

        def query11(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.contains("at")
            return maximum

        assert self.medrecord.query_nodes(query11) == "pat_5"

        def query12(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.is_string()
            return maximum

        assert self.medrecord.query_nodes(query12) == "pat_5"

        def query13(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.is_int()
            return maximum

        assert self.medrecord.query_nodes(query13) == 1

    def test_node_index_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((10, {}))
        self.medrecord.add_nodes((11, {}))

        def query1(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.add(2)
            return maximum

        assert self.medrecord.query_nodes(query1) == 13

        def query2(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.subtract(2)
            return maximum

        assert self.medrecord.query_nodes(query2) == 9

        def query3(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.multiply(2)
            return maximum

        assert self.medrecord.query_nodes(query3) == 22

        def query4(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.modulo(2)
            return maximum

        assert self.medrecord.query_nodes(query4) == 1

        def query5(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.power(2)
            return maximum

        assert self.medrecord.query_nodes(query5) == 121

        def query6(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.subtract(12)
            maximum.absolute()
            return maximum

        assert self.medrecord.query_nodes(query6) == 1

        self.medrecord.add_nodes((" Hello ", {}))

        def query7(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.trim()
            return maximum

        assert self.medrecord.query_nodes(query7) == "Hello"

        def query8(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.trim_start()
            return maximum

        assert self.medrecord.query_nodes(query8) == "Hello "

        def query9(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.trim_end()
            return maximum

        assert self.medrecord.query_nodes(query9) == " Hello"

        def query10(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.slice(0, 3)
            return maximum

        assert self.medrecord.query_nodes(query10) == " He"

        def query11(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.lowercase()
            return maximum

        assert self.medrecord.query_nodes(query11) == " hello "

        def query12(node: NodeOperand) -> NodeIndexOperand:
            node.index().contains(" ")
            maximum = node.index().max()
            maximum.uppercase()
            return maximum

        assert self.medrecord.query_nodes(query12) == " HELLO "

        def query13(node: NodeOperand) -> NodeIndexOperand:
            node.in_group("patient")
            maximum = node.index().max()
            maximum.either_or(
                lambda index: index.equal_to("not_a_node"),
                lambda index: index.equal_to("pat_5"),
            )
            return maximum

        assert self.medrecord.query_nodes(query13) == "pat_5"

        def query14(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            maximum.exclude(
                lambda index: index.is_string(),
            )
            return maximum

        assert self.medrecord.query_nodes(query14) == 11

        def query15(node: NodeOperand) -> NodeIndexOperand:
            node.index().is_int()
            maximum = node.index().max()
            clone = maximum.clone()
            maximum.add(10)
            return clone

        assert self.medrecord.query_nodes(query15) == 11


class TestEdgeIndicesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_indices_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().equal_to(0)
            return edge.index()

        assert self.medrecord.query_edges(query1) == [0]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().not_equal_to(0)
            return edge.index()

        assert self.medrecord.query_edges(query2) == [1]

        def query3(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            return edge.index()

        assert sorted(self.medrecord.query_edges(query3)) == [0, 1]

        def query4(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().is_not_in([1, 2])
            return edge.index()

        assert self.medrecord.query_edges(query4) == [0]

        def query5(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(1)
            return edge.index()

        assert self.medrecord.query_edges(query5) == [0]

        def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than_or_equal_to(1)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query6)) == [0, 1]

        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().greater_than(0)
            return edge.index()

        assert self.medrecord.query_edges(query7) == [1]

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().greater_than_or_equal_to(0)
            return edge.index()

        assert sorted(self.medrecord.query_edges(query8)) == [0, 1]

        def query9(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().starts_with(0)
            return edge.index()

        assert self.medrecord.query_edges(query9) == [0]

        def query10(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().ends_with(1)
            return edge.index()

        assert self.medrecord.query_edges(query10) == [1]

        def query11(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([0, 1])
            edge.index().contains(1)
            return edge.index()

        assert self.medrecord.query_edges(query11) == [1]

        def query12(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([0, 1])
            indices.is_max()
            return indices

        assert self.medrecord.query_edges(query12) == [1]

        def query13(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([0, 1])
            indices.is_min()
            return indices

        assert self.medrecord.query_edges(query13) == [0]

    def test_edge_indices_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([10, 11])
            return edge.index().max()

        assert self.medrecord.query_edges(query1) == 11

        def query2(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([10, 11])
            return edge.index().min()

        assert self.medrecord.query_edges(query2) == 10

        def query3(edge: EdgeOperand) -> EdgeIndexOperand:
            return edge.index().count()

        assert self.medrecord.query_edges(query3) == 160

        def query4(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([10, 11])
            return edge.index().sum()

        assert self.medrecord.query_edges(query4) == 21

        def query5(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([10, 11])
            edge.index().greater_than(10)
            return edge.index().random()

        assert self.medrecord.query_edges(query5) == 11

        def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([10, 11])
            indices.add(2)
            return indices

        assert sorted(self.medrecord.query_edges(query6)) == [12, 13]

        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([10, 11])
            indices.subtract(2)
            return indices

        assert sorted(self.medrecord.query_edges(query7)) == [8, 9]

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([10, 11])
            indices.multiply(2)
            return indices

        assert sorted(self.medrecord.query_edges(query8)) == [20, 22]

        def query9(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([10, 11])
            indices.modulo(2)
            return indices

        assert sorted(self.medrecord.query_edges(query9)) == [0, 1]

        def query10(edge: EdgeOperand) -> EdgeIndicesOperand:
            indices = edge.index()
            indices.is_in([10, 11])
            indices.power(2)
            return indices

        assert sorted(self.medrecord.query_edges(query10)) == [100, 121]

        def query11(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().either_or(
                lambda index: index.equal_to(10),
                lambda index: index.equal_to(11),
            )
            return edge.index()

        assert sorted(self.medrecord.query_edges(query11)) == [10, 11]

        def query12(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().exclude(lambda index: index.greater_than(1))
            return edge.index()

        assert sorted(self.medrecord.query_edges(query12)) == [0, 1]

        def query13(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().is_in([10, 11])
            clone = edge.index().clone()
            edge.index().add(10)
            return clone

        assert sorted(self.medrecord.query_edges(query13)) == [10, 11]


class TestEdgeIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_index_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.equal_to(0)
            return minimum

        assert self.medrecord.query_edges(query1) == 0

        def query2(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([0, 1])
            maximum = edge.index().max()
            maximum.not_equal_to(0)
            return maximum

        assert self.medrecord.query_edges(query2) == 1

        def query3(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.is_in([0, 1])
            return minimum

        assert self.medrecord.query_edges(query3) == 0

        def query4(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.is_not_in([1, 2])
            return minimum

        assert self.medrecord.query_edges(query4) == 0

        def query5(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.less_than(1)
            return minimum

        assert self.medrecord.query_edges(query5) == 0

        def query6(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.less_than_or_equal_to(1)
            return minimum

        assert self.medrecord.query_edges(query6) == 0

        def query7(edge: EdgeOperand) -> EdgeIndexOperand:
            edge.index().is_in([0, 1])
            maximum = edge.index().max()
            maximum.greater_than(0)
            return maximum

        assert self.medrecord.query_edges(query7) == 1

        def query8(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.greater_than_or_equal_to(0)
            return minimum

        assert self.medrecord.query_edges(query8) == 0

        def query9(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.starts_with(0)
            return minimum

        assert self.medrecord.query_edges(query9) == 0

        def query10(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.ends_with(0)
            return minimum

        assert self.medrecord.query_edges(query10) == 0

        def query11(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.contains(0)
            return minimum

        assert self.medrecord.query_edges(query11) == 0

    def test_edge_indices_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.add(2)
            return minimum

        assert self.medrecord.query_edges(query1) == 2

        def query2(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.subtract(2)
            return minimum

        # TODO(@JabobKrauskopf, #386): underflow on negative edge indices
        assert self.medrecord.query_edges(query2) == 4294967294

        def query3(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.add(11)
            minimum.multiply(2)
            return minimum

        assert self.medrecord.query_edges(query3) == 22

        def query4(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.add(11)
            minimum.modulo(2)
            return minimum

        assert self.medrecord.query_edges(query4) == 1

        def query5(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.add(11)
            minimum.power(2)
            return minimum

        assert self.medrecord.query_edges(query5) == 121

        def query6(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.either_or(
                lambda index: index.equal_to(0),
                lambda index: index.equal_to(1),
            )
            return minimum

        assert self.medrecord.query_edges(query6) == 0

        def query7(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            minimum.exclude(lambda index: index.greater_than(1))
            return minimum

        assert self.medrecord.query_edges(query7) == 0

        def query8(edge: EdgeOperand) -> EdgeIndexOperand:
            minimum = edge.index().min()
            clone = minimum.clone()
            minimum.add(10)
            return clone

        assert self.medrecord.query_edges(query8) == 0


class TestNodeAttributesTreeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_attributes_tree_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            return node.attributes().max()

        assert self.medrecord.query_nodes(query1) == {"pat_1": "gender"}

        def query2(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            return node.attributes().min()

        assert self.medrecord.query_nodes(query2) == {"pat_1": "age"}

        def query3(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            return node.attributes().count()

        assert self.medrecord.query_nodes(query3) == {"pat_1": 2}

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {1: "value1", 2: "value2"}))

        def query4(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(0)
            return node.attributes().sum()

        assert self.medrecord.query_nodes(query4) == {0: 3}

        def query5(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.equal_to(1)
            return attributes.random()

        assert self.medrecord.query_nodes(query5) == {0: 1}

        def query6(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.is_int()
            return attributes

        result = self.medrecord.query_nodes(query6)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [1, 2]}

        self.medrecord.add_nodes(
            ("new_node", {1: "value1", "string_attribute": "value2"})
        )

        def query7(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to("new_node")
            attributes = node.attributes()
            attributes.is_string()
            return attributes

        assert self.medrecord.query_nodes(query7) == {"new_node": ["string_attribute"]}

        def query8(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.is_in([1])
            return attributes

        assert self.medrecord.query_nodes(query8) == {0: [1]}

        def query9(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.is_not_in([1])
            return attributes

        assert self.medrecord.query_nodes(query9) == {0: [2]}

        def query10(node: NodeOperand) -> NodeAttributesTreeOperand:
            query_node(node)
            attributes = node.attributes()
            attributes.starts_with("ag")
            return attributes

        assert self.medrecord.query_nodes(query10) == {"pat_1": ["age"]}

        def query11(node: NodeOperand) -> NodeAttributesTreeOperand:
            query_node(node)
            attributes = node.attributes()
            attributes.ends_with("ge")
            return attributes

        assert self.medrecord.query_nodes(query11) == {"pat_1": ["age"]}

        def query12(node: NodeOperand) -> NodeAttributesTreeOperand:
            query_node(node)
            attributes = node.attributes()
            attributes.contains("ge")
            return attributes

        result = self.medrecord.query_nodes(query12)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {"pat_1": ["age", "gender"]}

        def query13(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.is_max()
            return attributes

        assert self.medrecord.query_nodes(query13) == {0: [2]}

        def query14(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.is_min()
            return attributes

        assert self.medrecord.query_nodes(query14) == {0: [1]}

        def query15(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.greater_than(1)
            return attributes

        assert self.medrecord.query_nodes(query15) == {0: [2]}

        def query16(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.greater_than_or_equal_to(1)
            return attributes

        result = self.medrecord.query_nodes(query16)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [1, 2]}

        def query17(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.less_than(2)
            return attributes

        assert self.medrecord.query_nodes(query17) == {0: [1]}

        def query18(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.less_than_or_equal_to(2)
            return attributes

        result = self.medrecord.query_nodes(query18)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [1, 2]}

        def query19(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.equal_to(1)
            return attributes

        assert self.medrecord.query_nodes(query19) == {0: [1]}

        def query20(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.not_equal_to(1)
            return attributes

        assert self.medrecord.query_nodes(query20) == {0: [2]}

    def test_node_attributes_tree_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))

        def query1(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.add(2)
            return attributes

        result = self.medrecord.query_nodes(query1)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [12, 13]}

        def query2(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.subtract(2)
            return attributes

        result = self.medrecord.query_nodes(query2)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [8, 9]}

        def query3(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.multiply(2)
            return attributes

        result = self.medrecord.query_nodes(query3)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [20, 22]}

        def query4(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.modulo(2)
            return attributes

        result = self.medrecord.query_nodes(query4)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [0, 1]}

        def query5(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.power(2)
            return attributes

        result = self.medrecord.query_nodes(query5)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [100, 121]}

        def query6(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.subtract(12)
            attributes.absolute()
            return attributes

        result = self.medrecord.query_nodes(query6)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [1, 2]}

        self.medrecord.add_nodes((1, {" Hello ": "value1"}))

        def query7(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.trim()
            return attributes

        assert self.medrecord.query_nodes(query7) == {1: ["Hello"]}

        def query8(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.trim_start()
            return attributes

        assert self.medrecord.query_nodes(query8) == {1: ["Hello "]}

        def query9(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.trim_end()
            return attributes

        assert self.medrecord.query_nodes(query9) == {1: [" Hello"]}

        def query10(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.slice(0, 3)
            return attributes

        assert self.medrecord.query_nodes(query10) == {1: [" He"]}

        def query11(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.lowercase()
            return attributes

        assert self.medrecord.query_nodes(query11) == {1: [" hello "]}

        def query12(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(1)
            attributes = node.attributes()
            attributes.uppercase()
            return attributes

        assert self.medrecord.query_nodes(query12) == {1: [" HELLO "]}

        def query13(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            attributes.either_or(
                lambda attribute: attribute.equal_to("not_a_node"),
                lambda attribute: attribute.contains(0),
            )
            return attributes

        assert self.medrecord.query_nodes(query13) == {0: [10]}

        def query14(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().is_int()
            attributes = node.attributes()
            attributes.exclude(
                lambda attribute: attribute.contains(0),
            )
            return attributes

        assert self.medrecord.query_nodes(query14) == {0: [11], 1: [" Hello "]}

        def query15(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.index().equal_to(0)
            attributes = node.attributes()
            clone = attributes.clone()
            attributes.add(10)
            return clone

        result = self.medrecord.query_nodes(query15)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {0: [10, 11]}


class TestEdgeAttributesTreeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        # Available nodes: pat_1, pat_2, pat_3, pat_4, pat_5 (and others from dataset)
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            [("pat_1", "pat_2", {1: "value1", 2: "value2"})]
        )  # Edge index 160
        self.medrecord.add_edges(
            [("pat_1", "pat_3", {10: "v1", 11: "v2"})]
        )  # Edge index 161
        self.medrecord.add_edges(
            [("pat_2", "pat_4", {" Hello ": "v3"})]
        )  # Edge index 162
        self.medrecord.add_edges(
            [("pat_3", "pat_5", {"string_attribute": "v4"})]
        )  # Edge index 163
        self.medrecord.add_edges(
            [("pat_4", "pat_1", {"a_attribute": "v5", "b_attribute": "v6"})]
        )  # Edge index 164

    def test_edge_attributes_tree_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().max()

        assert self.medrecord.query_edges(query1) == {164: "b_attribute"}

        def query2(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().min()

        assert self.medrecord.query_edges(query2) == {164: "a_attribute"}

        def query3(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().count()

        assert self.medrecord.query_edges(query3) == {164: 2}

        def query4(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 160)
            return edge.attributes().sum()

        assert self.medrecord.query_edges(query4) == {160: 3}

        def query5(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.equal_to(1)
            return attributes.random()

        assert self.medrecord.query_edges(query5) == {160: 1}

        def query6(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.is_int()
            return attributes

        result = self.medrecord.query_edges(query6)
        result = {key: sorted(value) for key, value in result.items()}

        assert result == {160: [1, 2]}

        def query7(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 163)
            attributes = edge.attributes()
            attributes.is_string()
            return attributes

        assert self.medrecord.query_edges(query7) == {163: ["string_attribute"]}

        def query8(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.is_in([1])
            return attributes

        assert self.medrecord.query_edges(query8) == {160: [1]}

        def query9(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.is_not_in([1])
            return attributes

        assert self.medrecord.query_edges(query9) == {160: [2]}

        def query10(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 164)
            attributes = edge.attributes()
            attributes.starts_with("a_")
            return attributes

        assert self.medrecord.query_edges(query10) == {164: ["a_attribute"]}

        def query11(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 164)
            attributes = edge.attributes()
            attributes.ends_with("attribute")
            return attributes

        result = self.medrecord.query_edges(query11)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {164: ["a_attribute", "b_attribute"]}

        def query12(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 164)
            attributes = edge.attributes()
            attributes.contains("_attribute")
            return attributes

        result = self.medrecord.query_edges(query12)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {164: ["a_attribute", "b_attribute"]}

        def query13(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.is_max()
            return attributes

        assert self.medrecord.query_edges(query13) == {160: [2]}

        def query14(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.is_min()
            return attributes

        assert self.medrecord.query_edges(query14) == {160: [1]}

        def query15(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.greater_than(1)
            return attributes

        assert self.medrecord.query_edges(query15) == {160: [2]}

        def query16(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.greater_than_or_equal_to(1)
            return attributes

        result = self.medrecord.query_edges(query16)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {160: [1, 2]}

        def query17(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.less_than(2)
            return attributes

        assert self.medrecord.query_edges(query17) == {160: [1]}

        def query18(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.less_than_or_equal_to(2)
            return attributes

        result = self.medrecord.query_edges(query18)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {160: [1, 2]}

        def query19(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.equal_to(1)
            return attributes

        assert self.medrecord.query_edges(query19) == {160: [1]}

        def query20(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 160)
            attributes = edge.attributes()
            attributes.not_equal_to(1)
            return attributes

        assert self.medrecord.query_edges(query20) == {160: [2]}

    def test_edge_attributes_tree_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.add(2)
            return attributes

        result = self.medrecord.query_edges(query1)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [12, 13]}

        def query2(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.subtract(2)
            return attributes

        result = self.medrecord.query_edges(query2)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [8, 9]}

        def query3(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.multiply(2)
            return attributes

        result = self.medrecord.query_edges(query3)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [20, 22]}

        def query4(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.modulo(2)
            return attributes

        result = self.medrecord.query_edges(query4)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [0, 1]}

        def query5(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.power(2)
            return attributes

        result = self.medrecord.query_edges(query5)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [100, 121]}

        def query6(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.subtract(12)
            attributes.absolute()
            return attributes

        result = self.medrecord.query_edges(query6)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [1, 2]}

        def query7(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.trim()
            return attributes

        assert self.medrecord.query_edges(query7) == {162: ["Hello"]}

        def query8(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.trim_start()
            return attributes

        assert self.medrecord.query_edges(query8) == {162: ["Hello "]}

        def query9(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.trim_end()
            return attributes

        assert self.medrecord.query_edges(query9) == {162: [" Hello"]}

        def query10(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.slice(0, 3)
            return attributes

        assert self.medrecord.query_edges(query10) == {162: [" He"]}

        def query11(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.lowercase()
            return attributes

        assert self.medrecord.query_edges(query11) == {162: [" hello "]}

        def query12(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 162)
            attributes = edge.attributes()
            attributes.uppercase()
            return attributes

        assert self.medrecord.query_edges(query12) == {162: [" HELLO "]}

        def query13(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            attributes.either_or(
                lambda attribute: attribute.equal_to(10),
                lambda attribute: attribute.equal_to(99),
            )
            return attributes

        assert self.medrecord.query_edges(query13) == {161: [10]}

        def query14(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            edge.index().is_in([161, 162])
            attributes = edge.attributes()
            attributes.exclude(
                lambda attribute: attribute.is_string(),
            )
            return attributes

        result = self.medrecord.query_edges(query14)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [10, 11], 162: []}

        def query15(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            query_specific_edge(edge, 161)
            attributes = edge.attributes()
            clone = attributes.clone()
            attributes.add(10)
            return clone

        result = self.medrecord.query_edges(query15)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [10, 11]}


class TestNodeMultipleAttributesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_multiple_attributes_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            return node.attributes().max().max()

        assert self.medrecord.query_nodes(query1) == (1, 13)

        def query2(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            return node.attributes().min().min()

        assert self.medrecord.query_nodes(query2) == (0, 10)

        def query3(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            return node.attributes().max().count()

        assert self.medrecord.query_nodes(query3) == 2

        def query4(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            return node.attributes().min().sum()

        assert self.medrecord.query_nodes(query4) == 22

        def query5(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attributes = node.attributes().max()
            attributes.equal_to(13)
            return attributes.random()

        assert self.medrecord.query_nodes(query5) == (1, 13)

        def query6(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query6) == {0: 11, 1: 13}

        def query7(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            attribute = node.attributes().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query7) == {"pat_1": "gender"}

        def query8(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_max()
            return attribute

        assert self.medrecord.query_nodes(query8) == {1: 13}

        def query9(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.is_min()
            return attribute

        assert self.medrecord.query_nodes(query9) == {0: 10}

        def query10(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.greater_than(12)
            return attribute

        assert self.medrecord.query_nodes(query10) == {1: 13}

        def query11(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_nodes(query11) == {0: 10, 1: 12}

        def query12(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_nodes(query12) == {0: 11}

        def query13(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query13) == {0: 10, 1: 12}

        def query14(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query14) == {0: 11}

        def query15(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query15) == {1: 13}

        def query16(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_in([11])
            return attribute

        assert self.medrecord.query_nodes(query16) == {0: 11}

        def query17(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_nodes(query17) == {1: 13}

        def query18(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            attribute = node.attributes().min()
            attribute.starts_with("a")
            return attribute

        assert self.medrecord.query_nodes(query18) == {"pat_1": "age"}

        def query19(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            attribute = node.attributes().min()
            attribute.ends_with("e")
            return attribute

        assert self.medrecord.query_nodes(query19) == {"pat_1": "age"}

        def query20(node: NodeOperand) -> NodeMultipleAttributesOperand:
            query_node(node)
            attribute = node.attributes().max()
            attribute.contains("ge")
            return attribute

        assert self.medrecord.query_nodes(query20) == {"pat_1": "gender"}

    def test_node_multiple_attributes_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query1) == {0: 13, 1: 15}

        def query2(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_nodes(query2) == {1: 11, 0: 9}

        def query3(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query3) == {0: 22, 1: 26}

        def query4(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query4) == {0: 1, 1: 1}

        def query5(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query5) == {0: 121, 1: 169}

        def query6(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.subtract(12)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query6) == {0: 1, 1: 1}

        def query7(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.either_or(
                lambda attribute: attribute.equal_to(13),
                lambda attribute: attribute.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query7) == {1: 13}

        def query8(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.exclude(
                lambda attribute: attribute.contains(13),
            )
            return attribute

        assert self.medrecord.query_nodes(query8) == {0: 11}

        self.medrecord.add_nodes((2, {" Hello ": "value1"}))

        def query9(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query9) == {2: "Hello"}

        def query10(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query10) == {2: "Hello "}

        def query11(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query11) == {2: " Hello"}

        def query12(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query12) == {2: " He"}

        def query13(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query13) == {2: " hello "}

        def query14(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query14) == {2: " HELLO "}

        def query15(node: NodeOperand) -> NodeMultipleValuesOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            return attribute.to_values()

        assert self.medrecord.query_nodes(query15) == {
            0: "value2",
            1: "value4",
            2: "value1",
        }

        def query16(node: NodeOperand) -> NodeMultipleAttributesOperand:
            node.index().is_in([0, 1])
            attribute = node.attributes().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_nodes(query16) == {0: 11, 1: 13}


class TestEdgeMultipleAttributesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()

        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_3", "pat_4", {12: "value3", 13: "value4"})])
        self.medrecord.add_edges([("pat_5", "pat_1", {" Hello ": "value5"})])

    def test_edge_multiple_attributes_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().max().max()

        assert self.medrecord.query_edges(query1) == (161, 13)

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().min().min()

        assert self.medrecord.query_edges(query2) == (160, 10)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().max().count()

        assert self.medrecord.query_edges(query3) == 2

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().min().sum()

        assert self.medrecord.query_edges(query4) == 22

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attributes = edge.attributes().max()
            attributes.equal_to(13)
            return attributes.random()

        assert self.medrecord.query_edges(query5) == (161, 13)

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query6) == {160: 11, 161: 13}

        def query7(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query7) == {0: "time"}

        def query8(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_max()
            return attribute

        assert self.medrecord.query_edges(query8) == {161: 13}

        def query9(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.is_min()
            return attribute

        assert self.medrecord.query_edges(query9) == {160: 10}

        def query10(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.greater_than(12)
            return attribute

        assert self.medrecord.query_edges(query10) == {161: 13}

        def query11(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_edges(query11) == {160: 10, 161: 12}

        def query12(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_edges(query12) == {160: 11}

        def query13(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query13) == {160: 10, 161: 12}

        def query14(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query14) == {160: 11}

        def query15(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query15) == {161: 13}

        def query16(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_in([11, 12])
            return attribute

        assert self.medrecord.query_edges(query16) == {160: 11}

        def query17(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_not_in([11, 12])
            return attribute

        assert self.medrecord.query_edges(query17) == {161: 13}

        def query18(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().min()
            attribute.starts_with("dur")
            return attribute

        assert self.medrecord.query_edges(query18) == {0: "duration_days"}

        def query19(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().min()
            attribute.ends_with("ys")
            return attribute

        assert self.medrecord.query_edges(query19) == {0: "duration_days"}

        def query20(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max()
            attribute.contains("im")
            return attribute

        assert self.medrecord.query_edges(query20) == {0: "time"}

    def test_edge_multiple_attributes_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query1) == {160: 13, 161: 15}

        def query2(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_edges(query2) == {160: 9, 161: 11}

        def query3(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query3) == {160: 22, 161: 26}

        def query4(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query4) == {160: 1, 161: 1}

        def query5(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query5) == {160: 121, 161: 169}

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.subtract(12)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query6) == {160: 1, 161: 1}

        def query7(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.either_or(
                lambda attr: attr.equal_to(13),
                lambda attr: attr.equal_to(10),
            )
            return attribute

        assert self.medrecord.query_edges(query7) == {161: 13}

        def query8(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.exclude(
                lambda attr: attr.equal_to(13),
            )
            return attribute

        assert self.medrecord.query_edges(query8) == {160: 11}

        def query9(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query9) == {162: "Hello"}

        def query10(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query10) == {162: "Hello "}

        def query11(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query11) == {162: " Hello"}

        def query12(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query12) == {162: " He"}

        def query13(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query13) == {162: " hello "}

        def query14(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query14) == {162: " HELLO "}

        def query15(edge: EdgeOperand) -> EdgeMultipleValuesOperand:
            edge.index().is_in([160, 161, 162])
            attribute = edge.attributes().max()
            return attribute.to_values()

        assert self.medrecord.query_edges(query15) == {
            160: "value2",
            161: "value4",
            162: "value5",
        }

        def query16(edge: EdgeOperand) -> EdgeMultipleAttributesOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_edges(query16) == {160: 11, 161: 13}


class TestNodeSingleAttributeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_single_attribute_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query1) == ("pat_1", "gender")

        def query2(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query2) == (0, 10)

        def query3(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_nodes(query3) == (1, 13)

        def query4(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_nodes(query4) == (0, 10)

        def query5(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_nodes(query5) == (0, 10)

        def query6(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query6) == (0, 10)

        def query7(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.equal_to(13)
            return attribute

        assert self.medrecord.query_nodes(query7) == (1, 13)

        def query8(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query8) == (1, 13)

        def query9(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.is_in([13])
            return attribute

        assert self.medrecord.query_nodes(query9) == (1, 13)

        def query10(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_nodes(query10) == (1, 13)

        def query11(node: NodeOperand) -> NodeSingleAttributeOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.starts_with("g")
            return attribute

        assert self.medrecord.query_nodes(query11) == ("pat_1", "gender")

        def query12(node: NodeOperand) -> NodeSingleAttributeOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.ends_with("er")
            return attribute

        assert self.medrecord.query_nodes(query12) == ("pat_1", "gender")

        def query13(node: NodeOperand) -> NodeSingleAttributeOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.contains("ge")
            return attribute

        assert self.medrecord.query_nodes(query13) == ("pat_1", "gender")

        self.medrecord.add_nodes((2, {" Hello ": "value1"}))

        def query14(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query14) == (2, "Hello")

        def query15(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query15) == (2, "Hello ")

        def query16(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query16) == (2, " Hello")

        def query17(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query17) == (2, " He")

        def query18(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query18) == (2, " hello ")

        def query19(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query19) == (2, " HELLO ")

    def test_single_attribute_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query1) == (1, 15)

        def query2(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_nodes(query2) == (1, 11)

        def query3(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query3) == (0, 20)

        def query4(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query4) == (1, 1)

        def query5(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query5) == (1, 169)

        def query6(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.subtract(14)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query6) == (1, 1)

        def query7(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.either_or(
                lambda attribute: attribute.equal_to(13),
                lambda attribute: attribute.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query7) == (1, 13)

        def query8(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.exclude(
                lambda attribute: attribute.contains(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query8) == (1, 13)

        def query9(node: NodeOperand) -> NodeSingleAttributeOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_nodes(query9) == (1, 13)


class TestEdgeSingleAttributeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_3", "pat_4", {12: "value3", 13: "value4"})])
        self.medrecord.add_edges([("pat_5", "pat_1", {" Hello ": "value5"})])

    def test_single_attribute_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()

        def query1(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query1) == (0, "time")

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query2) == (160, 10)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_edges(query3) == (161, 13)

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_edges(query4) == (160, 10)

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_edges(query5) == (160, 10)

        def query6(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query6) == (160, 10)

        def query7(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.equal_to(13)
            return attribute

        assert self.medrecord.query_edges(query7) == (161, 13)

        def query8(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query8) == (161, 13)

        def query9(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.is_in([13])
            return attribute

        assert self.medrecord.query_edges(query9) == (161, 13)

        def query10(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_edges(query10) == (161, 13)

        def query11(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.starts_with("t")
            return attribute

        assert self.medrecord.query_edges(query11) == (0, "time")

        def query12(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.ends_with("me")
            return attribute

        assert self.medrecord.query_edges(query12) == (0, "time")

        def query13(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.contains("im")
            return attribute

        assert self.medrecord.query_edges(query13) == (0, "time")

        def query14(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query14) == (162, "Hello")

        def query15(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query15) == (162, "Hello ")

        def query16(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query16) == (162, " Hello")

        def query17(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query17) == (162, " He")

        def query18(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query18) == (162, " hello ")

        def query19(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query19) == (162, " HELLO ")

    def test_single_attribute_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query1) == (161, 15)

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_edges(query2) == (161, 11)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query3) == (160, 20)

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query4) == (161, 1)

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query5) == (161, 169)

        def query6(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.subtract(14)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query6) == (161, 1)

        def query7(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.either_or(
                lambda attr: attr.equal_to(13),
                lambda attr: attr.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_edges(query7) == (161, 13)

        def query8(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.exclude(
                lambda attr: attr.contains(12),
            )
            return attribute

        assert self.medrecord.query_edges(query8) == (161, 13)

        def query9(edge: EdgeOperand) -> EdgeSingleAttributeOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_edges(query9) == (161, 13)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestPythonTypesConversion)
    )
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeOperand))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestNodeMultipleValuesOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeMultipleValuesOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestNodeSingleValueOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeSingleValueOperand)
    )
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeIndicesOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeIndexOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeIndicesOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeIndexOperand))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestNodeAttributesTreeOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeAttributesTreeOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestNodeMultipleAttributesOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeMultipleAttributesOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestNodeSingleAttributeOperand)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeSingleAttributeOperand)
    )

    unittest.TextTestRunner(verbosity=2).run(suite)

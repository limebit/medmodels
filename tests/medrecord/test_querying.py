import operator
import unittest
from datetime import datetime, timedelta

import pytest

from medmodels import MedRecord
from medmodels.medrecord import EdgeIndex, NodeIndex
from medmodels.medrecord.querying import (
    EdgeAttributesTreeGroupOperand,
    EdgeAttributesTreeOperand,
    EdgeAttributesTreeQueryResult,
    EdgeDirection,
    EdgeIndexGroupOperand,
    EdgeIndexOperand,
    EdgeIndicesGroupOperand,
    EdgeIndicesOperand,
    EdgeIndicesQueryResult,
    EdgeMultipleAttributesWithIndexGroupOperand,
    EdgeMultipleAttributesWithIndexOperand,
    EdgeMultipleAttributesWithIndexQueryResult,
    EdgeMultipleAttributesWithoutIndexOperand,
    EdgeMultipleValuesWithIndexGroupOperand,
    EdgeMultipleValuesWithIndexOperand,
    EdgeMultipleValuesWithIndexQueryResult,
    EdgeMultipleValuesWithoutIndexOperand,
    EdgeOperand,
    EdgeOperandGroupDiscriminator,
    EdgeSingleAttributeWithIndexGroupOperand,
    EdgeSingleAttributeWithIndexOperand,
    EdgeSingleAttributeWithoutIndexGroupOperand,
    EdgeSingleAttributeWithoutIndexOperand,
    EdgeSingleValueWithIndexGroupOperand,
    EdgeSingleValueWithIndexGroupQueryResult,
    EdgeSingleValueWithIndexOperand,
    EdgeSingleValueWithoutIndexGroupOperand,
    EdgeSingleValueWithoutIndexGroupQueryResult,
    EdgeSingleValueWithoutIndexOperand,
    GroupKey,
    MatchMode,
    NodeAttributesTreeGroupOperand,
    NodeAttributesTreeOperand,
    NodeAttributesTreeQueryResult,
    NodeIndexGroupOperand,
    NodeIndexOperand,
    NodeIndicesGroupOperand,
    NodeIndicesOperand,
    NodeIndicesQueryResult,
    NodeMultipleAttributesWithIndexGroupOperand,
    NodeMultipleAttributesWithIndexOperand,
    NodeMultipleAttributesWithIndexQueryResult,
    NodeMultipleAttributesWithoutIndexOperand,
    NodeMultipleValuesWithIndexGroupOperand,
    NodeMultipleValuesWithIndexOperand,
    NodeMultipleValuesWithIndexQueryResult,
    NodeMultipleValuesWithoutIndexOperand,
    NodeOperand,
    NodeOperandGroupDiscriminator,
    NodeSingleAttributeWithIndexGroupOperand,
    NodeSingleAttributeWithIndexOperand,
    NodeSingleAttributeWithoutIndexGroupOperand,
    NodeSingleAttributeWithoutIndexOperand,
    NodeSingleValueWithIndexGroupOperand,
    NodeSingleValueWithIndexOperand,
    NodeSingleValueWithIndexQueryResult,
    NodeSingleValueWithoutIndexGroupOperand,
    NodeSingleValueWithoutIndexOperand,
    NodeSingleValueWithoutIndexQueryResult,
    PyEdgeIndexOperand,
    PyEdgeIndicesOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    PyEdgeMultipleValuesWithoutIndexOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeSingleValueWithoutIndexOperand,
    PyNodeIndexOperand,
    PyNodeIndicesOperand,
    PyNodeMultipleAttributesWithIndexOperand,
    PyNodeMultipleAttributesWithoutIndexOperand,
    PyNodeMultipleValuesWithIndexOperand,
    PyNodeMultipleValuesWithoutIndexOperand,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeSingleValueWithIndexOperand,
    PyNodeSingleValueWithoutIndexOperand,
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


def query_edge(edge: EdgeOperand, index: EdgeIndex = 0) -> None:
    edge.index().equal_to(index)


def query_specific_edge(
    edge: EdgeOperand, indices: EdgeIndex | list[EdgeIndex]
) -> None:
    if not isinstance(indices, list):
        indices = [indices]
    edge.index().is_in(indices)


class TestPythonTypesConversion(unittest.TestCase):
    def test_node_python_types_conversion(self) -> None:
        medrecord = MedRecord.from_simple_example_dataset()

        cache1: NodeSingleValueWithIndexOperand

        def query1(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
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
        assert isinstance(type2, PyNodeSingleValueWithIndexOperand)

        cache2: NodeMultipleValuesWithIndexOperand

        def query2(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
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
        assert isinstance(type4, PyNodeMultipleValuesWithIndexOperand)

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

        cache5: NodeSingleAttributeWithIndexOperand

        def query5(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
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
        assert isinstance(type10, PyNodeSingleAttributeWithIndexOperand)

        cache8: NodeMultipleAttributesWithIndexOperand

        def query8(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
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
        assert isinstance(type16, PyNodeMultipleAttributesWithIndexOperand)

        cache9: NodeSingleValueWithoutIndexOperand

        def query9(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            nonlocal cache9
            cache9 = node.attribute("age").mean()
            return cache9

        medrecord.query_nodes(query9)

        type17 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                1.0
            )
        )
        type18 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                cache9  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type17, MedRecordValue)
        assert isinstance(type18, PyNodeSingleValueWithoutIndexOperand)

        cache10: NodeMultipleValuesWithoutIndexOperand

        def query10(
            node: NodeOperand,
        ) -> NodeMultipleValuesWithoutIndexOperand:
            nonlocal cache10
            cache10 = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("age"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return cache10

        medrecord.query_nodes(query10)

        type19 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            cache10  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type19, PyNodeMultipleValuesWithoutIndexOperand)

        cache11: NodeSingleAttributeWithoutIndexOperand

        def query11(
            node: NodeOperand,
        ) -> NodeSingleAttributeWithoutIndexOperand:
            nonlocal cache11
            cache11 = node.attributes().count().count()
            return cache11

        medrecord.query_nodes(query11)

        type20 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            cache11  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type20, PyNodeSingleAttributeWithoutIndexOperand)

        cache12: NodeMultipleAttributesWithoutIndexOperand

        def query12(
            node: NodeOperand,
        ) -> NodeMultipleAttributesWithoutIndexOperand:
            nonlocal cache12
            cache12 = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("age"))
                .attributes()
                .count()
                .count()
                .ungroup()
            )
            return cache12

        medrecord.query_nodes(query12)

        type21 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            cache12  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )
        assert isinstance(type21, PyNodeMultipleAttributesWithoutIndexOperand)

    def test_edge_python_types_conversion(self) -> None:
        medrecord = MedRecord.from_simple_example_dataset()

        cache1: EdgeSingleValueWithIndexOperand

        def query1(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            nonlocal cache1
            cache1 = edge.attribute("time").max()
            return cache1

        medrecord.query_edges(query1)

        type1 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                cache1  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type1, PyEdgeSingleValueWithIndexOperand)

        cache2: EdgeMultipleValuesWithIndexOperand

        def query2(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            nonlocal cache2
            cache2 = edge.attribute("time")
            return cache2

        medrecord.query_edges(query2)

        type2 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            cache2  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type2, PyEdgeMultipleValuesWithIndexOperand)

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

        cache5: EdgeSingleAttributeWithIndexOperand

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            nonlocal cache5
            cache5 = edge.attributes().max().max()
            return cache5

        medrecord.query_edges(query5)

        type6 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            cache5  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type6, PyEdgeSingleAttributeWithIndexOperand)

        cache6: EdgeMultipleAttributesWithIndexOperand

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            nonlocal cache6
            cache6 = edge.attributes().max()
            return cache6

        medrecord.query_edges(query6)
        type7 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            cache6  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )
        assert isinstance(type7, PyEdgeMultipleAttributesWithIndexOperand)

        cache7: EdgeSingleValueWithoutIndexOperand

        def query7(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            nonlocal cache7
            cache7 = edge.attribute("age").mean()
            return cache7

        medrecord.query_edges(query7)

        type8 = (
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                cache7  # noqa: F821 # pyright: ignore[reportUnboundVariable]
            )
        )

        assert isinstance(type8, PyEdgeSingleValueWithoutIndexOperand)

        cache8: EdgeMultipleValuesWithoutIndexOperand

        def query10(
            edge: EdgeOperand,
        ) -> EdgeMultipleValuesWithoutIndexOperand:
            nonlocal cache8
            cache8 = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("age")
                .mean()
                .ungroup()
            )
            return cache8

        medrecord.query_edges(query10)

        type9 = _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
            cache8  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type9, PyEdgeMultipleValuesWithoutIndexOperand)

        cache9: EdgeSingleAttributeWithoutIndexOperand

        def query11(
            edge: EdgeOperand,
        ) -> EdgeSingleAttributeWithoutIndexOperand:
            nonlocal cache9
            cache9 = edge.attributes().count().count()
            return cache9

        medrecord.query_edges(query11)

        type10 = _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
            cache9  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )

        assert isinstance(type10, PyEdgeSingleAttributeWithoutIndexOperand)

        cache10: EdgeMultipleAttributesWithoutIndexOperand

        def query12(
            edge: EdgeOperand,
        ) -> EdgeMultipleAttributesWithoutIndexOperand:
            nonlocal cache10
            cache10 = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attributes()
                .count()
                .count()
                .ungroup()
            )
            return cache10

        medrecord.query_edges(query12)

        type11 = _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
            cache10  # noqa: F821 # pyright: ignore[reportUnboundVariable]
        )
        assert isinstance(type11, PyEdgeMultipleAttributesWithoutIndexOperand)


class TestNodeOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_operand_attribute_simple(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
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

        self.medrecord.add_group("test_group", "diagnosis_10509002")

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group((["patient", "test_group"], MatchMode.ANY))  # Must be in ANY
            return node.index()

        assert sorted(self.medrecord.query_nodes(query3)) == [
            "diagnosis_10509002",
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

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

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.has_attribute((["gender", "age"], MatchMode.ANY))
            return node.index()

        assert self.medrecord.query_nodes(query3) == ["pat_1"]

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


class TestNodeGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_group_operand_attribute(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            return node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeMultipleValuesWithIndexQueryResult],
        ) -> tuple[GroupKey, NodeMultipleValuesWithIndexQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (
                value,
                dict(sorted(nodes_with_attributes.items(), key=operator.itemgetter(1))),
            )

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 22, "pat_3": 96}),
            ("M", {"pat_4": 19, "pat_5": 37, "pat_1": 42}),
        ]

    def test_group_operand_attributes(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.has_attribute("gender")
            return node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeAttributesTreeQueryResult],
        ) -> tuple[GroupKey, NodeAttributesTreeQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            sorted_nodes = sorted(
                ((k, sorted(v)) for k, v in dict(nodes_with_attributes).items()),
                key=operator.itemgetter(0),
            )
            return (value, dict(sorted_nodes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": ["age", "gender"], "pat_3": ["age", "gender"]}),
            (
                "M",
                {
                    "pat_1": ["age", "gender"],
                    "pat_4": ["age", "gender"],
                    "pat_5": ["age", "gender"],
                },
            ),
        ]

    def test_group_operand_index(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.has_attribute("gender")
            return node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

    def test_group_operand_in_group(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.in_group("patient")
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes_to_group("diagnosis", "pat_1")

        def query2(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.in_group((["patient", "diagnosis"], MatchMode.ALL))  # BOTH groups
            return group.index()

        result2 = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query2)),
            key=operator.itemgetter(0),
        )
        assert result2 == [
            ("M", ["pat_1"]),
        ]

    def test_group_operand_has_attribute(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.has_attribute("age")

            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

        def query2(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.has_attribute((["gender", "age"], MatchMode.ANY))

            return group.index()

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query2)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

    def test_group_operand_edges(self) -> None:
        def query(node: NodeOperand) -> EdgeIndicesGroupOperand:
            node.has_attribute("gender")
            edges = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).edges()
            edges.index().equal_to(0)
            return edges.index()

        result = sorted(self.medrecord.query_nodes(query), key=operator.itemgetter(0))
        assert result == [("F", []), ("M", [0])]

    def test_group_operand_neighbors(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(("pat_2", "pat_1", {}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.neighbors().in_group("patient")

            return group.index()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", ["pat_2"]),
            ("M", ["pat_1"]),
        ]

    def test_group_operand_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.either_or(
                lambda node: node.attribute("age").greater_than(90),
                lambda node: node.attribute("age").less_than(20),
            )
            return group.index()

        result = sorted(self.medrecord.query_nodes(query), key=operator.itemgetter(0))

        assert result == [
            ("F", ["pat_3"]),
            ("M", ["pat_4"]),
        ]

    def test_group_operand_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group.exclude(lambda node: node.attribute("age").less_than(90))
            group.in_group("patient")
            return group.index()

        result = sorted(self.medrecord.query_nodes(query), key=operator.itemgetter(0))

        assert result == [
            ("F", ["pat_3"]),
        ]

    def test_group_operand_clone(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
            group_clone = group.clone()
            group.attribute("age").greater_than(30)
            return group_clone.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]


class TestEdgeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_operand_attribute_simple(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
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

        self.medrecord.add_group("test_group", None, 1)

        def query3(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.either_or(query_edge, lambda edge: edge.index().equal_to(1))
            edge.in_group(
                (["patient_diagnosis", "test_group"], MatchMode.ANY)
            )  # Must be in ANY
            return edge.index()

        assert sorted(self.medrecord.query_edges(query3)) == [0, 1]

    def test_edge_operand_has_attribute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.has_attribute("time")
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.has_attribute((["time"], MatchMode.ANY))
            return edge.index()

        assert self.medrecord.query_edges(query2) == [0]

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


class TestEdgeGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_group_operand_attribute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("time")
                .min()
            )

        assert sorted(self.medrecord.query_edges(query)) == [
            ("pat_1", (0, datetime(2014, 2, 6, 0, 0))),
            ("pat_2", (66, datetime(2014, 8, 12, 9, 1, 28))),
            ("pat_3", (24, datetime(1962, 10, 21, 0, 0))),
            ("pat_4", (39, datetime(2014, 2, 26, 0, 0))),
            ("pat_5", (46, datetime(2004, 10, 22, 0, 0))),
        ]

    def test_edge_group_operand_attributes(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            return group.attributes()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeAttributesTreeQueryResult],
        ) -> tuple[GroupKey, EdgeAttributesTreeQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            sorted_nodes = sorted(
                ((k, sorted(v)) for k, v in dict(nodes_with_attributes).items()),
                key=operator.itemgetter(0),
            )
            return (value, dict(sorted_nodes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            (
                "pat_1",
                {
                    0: ["duration_days", "time"],
                    1: ["duration_days", "time"],
                    2: ["duration_days", "time"],
                    3: ["duration_days", "time"],
                    4: ["duration_days", "time"],
                },
            )
        ]

    def test_edge_group_operand_index(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 4]),
        ]

    def test_edge_group_operand_in_group(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.in_group("patient_diagnosis")
            group.index().less_than(5)
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 4]),
        ]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_group("temp_group", None, 0)

        def query2(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.in_group(
                (["patient_diagnosis", "temp_group"], MatchMode.ALL)
            )  # BOTH groups
            return group.index()

        result2 = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query2)),
            key=operator.itemgetter(0),
        )
        assert result2 == [
            ("pat_1", [0]),
        ]

    def test_edge_group_operand_has_attribute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.has_attribute("time")
            group.index().less_than(5)
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 4]),
        ]

        def query2(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.has_attribute((["time", "duration_days"], MatchMode.ANY))
            group.index().less_than(5)
            return group.index()

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query2)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 4]),
        ]

    def test_edge_group_operand_source_node(self) -> None:
        def query(edge: EdgeOperand) -> NodeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.TargetNode())
            group.index().less_than(5)
            return group.source_node().index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("diagnosis_195662009", ["pat_1"]),
            ("diagnosis_314529007", ["pat_1"]),
            ("diagnosis_73595000", ["pat_1"]),
            ("diagnosis_741062008", ["pat_1"]),
            ("diagnosis_82423001", ["pat_1"]),
        ]

    def test_edge_group_operand_target_node(self) -> None:
        def query(edge: EdgeOperand) -> NodeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            return group.target_node().index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
        ) -> tuple[GroupKey, NodeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            (
                "pat_1",
                [
                    "diagnosis_195662009",
                    "diagnosis_314529007",
                    "diagnosis_73595000",
                    "diagnosis_741062008",
                    "diagnosis_82423001",
                ],
            )
        ]

    def test_edge_group_operand_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.either_or(
                lambda edge: edge.index().less_than(4),
                lambda edge: edge.index().equal_to(10),
            )
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 10]),
        ]

    def test_edge_group_operand_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.exclude(lambda edge: edge.index().greater_than(1))
            group.in_group("patient_diagnosis")
            return group.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("pat_1", [0, 1])]

    def test_edge_group_operand_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            clone = group.clone()
            group.attribute("time").less_than(datetime(2025, 1, 1))
            return clone.index()

        def sort_tuple(
            tuple_to_sort: tuple[GroupKey, EdgeIndicesQueryResult],
        ) -> tuple[GroupKey, EdgeIndicesQueryResult]:
            value, nodes_with_attributes = tuple_to_sort
            return (value, sorted(nodes_with_attributes))

        result = sorted(
            (sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", [0, 1, 2, 3, 4]),
        ]


class TestNodeMultipleValuesWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def sort_tuple(
        self,
        tuple_to_sort: tuple[GroupKey, NodeMultipleValuesWithIndexQueryResult],
    ) -> tuple[GroupKey, NodeMultipleValuesWithIndexQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        return (
            value,
            dict(sorted(nodes_with_attributes.items(), key=operator.itemgetter(1))),
        )

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.max()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96)),
            ("M", ("pat_1", 42)),
        ]

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.min()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_2", 22)),
            ("M", ("pat_4", 19)),
        ]

    def test_mean(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.mean()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 59.0),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_median(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.median()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 59.0),
            ("M", 37.0),
        ]

    def test_mode(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.less_than(25)
            return group.mode()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 22),
            ("M", 19),
        ]

    def test_std(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.std()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 37.0),
            ("M", pytest.approx(9.877, rel=1e-2)),
        ]

    def test_var(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.var()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 1369.0),
            ("M", pytest.approx(97.55, rel=1e-2)),
        ]

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.count()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 2),
            ("M", 3),
        ]

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.sum()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", 118),
            ("M", 98),
        ]

    def test_random(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.greater_than(40)
            return group.random()

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96)),
            ("M", ("pat_1", 42)),
        ]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("gender")
            group.is_string()
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", {"pat_2": "F", "pat_3": "F"}),
            ("M", {"pat_1": "M", "pat_4": "M", "pat_5": "M"}),
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.is_int()
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", {"pat_2": 22, "pat_3": 96}),
            ("M", {"pat_4": 19, "pat_5": 37, "pat_1": 42}),
        ]

    def test_is_float(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"float_attribute": 2.3, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("float_attribute")
            group.is_float()
            return group

        assert self.medrecord.query_nodes(query) == [("M", {"pat_6": 2.3})]

    def test_is_bool(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"bool_attribute": True, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("bool_attribute")
            group.is_bool()
            return group

        assert self.medrecord.query_nodes(query) == [("M", {"pat_6": True})]

    def test_is_datetime(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"datetime_attribute": datetime(2023, 10, 1), "gender": "M"})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("datetime_attribute")
            group.is_datetime()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", {"pat_6": datetime(2023, 10, 1)})
        ]

    def test_is_duration(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"duration_attribute": timedelta(days=5), "gender": "M"})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("duration_attribute")
            group.is_duration()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", {"pat_6": timedelta(days=5)})
        ]

    def test_is_null(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"null_attribute": None, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("null_attribute")
            group.is_null()
            return group

        assert self.medrecord.query_nodes(query) == [("M", {"pat_6": None})]

    def test_is_max(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.is_max()
            return group

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_1": 42}),
        ]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.is_min()
            return group

        assert sorted(self.medrecord.query_nodes(query)) == [
            ("F", {"pat_2": 22}),
            ("M", {"pat_4": 19}),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.greater_than(40)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_1": 42}),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.greater_than_or_equal_to(42)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_1": 42}),
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.equal_to(42)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {}),
            ("M", {"pat_1": 42}),
        ]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.less_than(22)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {}),
            ("M", {"pat_4": 19}),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.less_than_or_equal_to(22)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 22}),
            ("M", {"pat_4": 19}),
        ]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.not_equal_to(22)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_4": 19, "pat_5": 37, "pat_1": 42}),
        ]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.is_in([19, 22, 42])
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 22}),
            ("M", {"pat_4": 19, "pat_1": 42}),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.is_not_in([19, 22, 42])
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_5": 37}),
        ]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.starts_with(1)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {}),
            ("M", {"pat_4": 19}),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.ends_with(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 22}),
            ("M", {"pat_1": 42}),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 22}),
            ("M", {"pat_1": 42}),
        ]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            group.add(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 24}),
            ("M", {"pat_1": 44}),
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            group.subtract(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 20}),
            ("M", {"pat_1": 40}),
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            group.subtract(2)
            group.multiply(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 40}),
            ("M", {"pat_1": 80}),
        ]

    def test_divide(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            group.subtract(2)
            group.divide(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 10}),
            ("M", {"pat_1": 20}),
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.contains(2)
            group.subtract(2)
            group.power(2)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 400}),
            ("M", {"pat_1": 1600}),
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.greater_than(30)
            group.modulo(5)
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 1}),
            ("M", {"pat_5": 2, "pat_1": 2}),
        ]

    def test_round_ceil_floor(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"float_attribute": 2.34567, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("float_attribute")
            group.round()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", {"pat_6": 2}),
        ]

        def query_ceil(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("float_attribute")
            group.ceil()
            return group

        assert self.medrecord.query_nodes(query_ceil) == [
            ("M", {"pat_6": 3}),
        ]

        def query_floor(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("float_attribute")
            group.floor()
            return group

        assert self.medrecord.query_nodes(query_floor) == [
            ("M", {"pat_6": 2}),
        ]

    def test_absolute(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"float_attribute": -2.35, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("float_attribute")
            group.absolute()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", {"pat_6": 2.35}),
        ]

    def test_sqrt(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"number": 81, "gender": "M"}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("number")
            group.sqrt()
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("M", {"pat_6": 9.0}),
        ]

    def test_string_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"string_attribute": " Hello World ", "gender": "M"})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.trim()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", {"pat_6": "Hello World"}),
        ]

        def query_trim_start(
            node: NodeOperand,
        ) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.trim_start()
            return group

        assert self.medrecord.query_nodes(query_trim_start) == [
            ("M", {"pat_6": "Hello World "}),
        ]

        def query_trim_end(
            node: NodeOperand,
        ) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.trim_end()
            return group

        assert self.medrecord.query_nodes(query_trim_end) == [
            ("M", {"pat_6": " Hello World"}),
        ]

        def query_lowercase(
            node: NodeOperand,
        ) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.lowercase()
            return group

        assert self.medrecord.query_nodes(query_lowercase) == [
            ("M", {"pat_6": " hello world "}),
        ]

        def query_uppercase(
            node: NodeOperand,
        ) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.uppercase()
            return group

        assert self.medrecord.query_nodes(query_uppercase) == [
            ("M", {"pat_6": " HELLO WORLD "}),
        ]

        def query_slice(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("string_attribute")
            group.slice(0, 6)
            return group

        assert self.medrecord.query_nodes(query_slice) == [
            ("M", {"pat_6": " Hello"}),
        ]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.either_or(
                lambda group: group.less_than(20),
                lambda group: group.greater_than(90),
            )
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_4": 19}),
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            group.exclude(lambda group: group.less_than(30))
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_3": 96}),
            ("M", {"pat_5": 37, "pat_1": 42}),
        ]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            return group.ungroup()

        result = dict(
            sorted(
                self.medrecord.query_nodes(query).items(), key=operator.itemgetter(0)
            )
        )
        assert result == {
            "pat_1": 42,
            "pat_2": 22,
            "pat_3": 96,
            "pat_4": 19,
            "pat_5": 37,
        }

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute("gender")
            group = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attribute("age")
            clone = group.clone()
            group.less_than(30)
            return clone

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", {"pat_2": 22, "pat_3": 96}),
            ("M", {"pat_4": 19, "pat_5": 37, "pat_1": 42}),
        ]


class TestNodeMultipleValuesWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            node.has_attribute("gender")
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.max()

        assert self.medrecord.query_nodes(query) == 59

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            node.has_attribute("gender")
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.min()

        assert self.medrecord.query_nodes(query) == pytest.approx(32.66, rel=1e-2)

    def test_mean(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            node.has_attribute("gender")
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.mean()

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_median(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            node.has_attribute("gender")
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.median()

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_mode(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            node.has_attribute("gender")
            node.attribute("age").less_than(25)
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.mode()

        assert self.medrecord.query_nodes(query) in [22, 19]

    def test_std(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.std()

        std = self.medrecord.query_nodes(query)
        assert std == pytest.approx(13.166, rel=1e-2)

    def test_var(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.var()

        var = self.medrecord.query_nodes(query)
        assert var == pytest.approx(173.361, rel=1e-2)

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.count()

        assert self.medrecord.query_nodes(query) == 2

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.sum()

        assert self.medrecord.query_nodes(query) == pytest.approx(91.66, rel=1e-2)

    def test_random(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            return values.random()

        assert self.medrecord.query_nodes(query) in [
            pytest.approx(32.66, rel=1e-2),
            59,
        ]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("gender")
                .mode()
                .ungroup()
            )
            values.is_string()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == ["F", "M"]

    def test_is_int(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"gender": "F", "int_attribute": 5}))
        self.medrecord.add_nodes(("pat_7", {"gender": "M", "int_attribute": 10}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("int_attribute")
                .mode()
                .ungroup()
            )
            values.is_int()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [5, 10]

    def test_is_float(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.is_float()
            return values.count()

        assert self.medrecord.query_nodes(query) == 2

    def test_is_bool(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"gender": "F", "bool_attribute": True}))
        self.medrecord.add_nodes(("pat_7", {"gender": "M", "bool_attribute": False}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("bool_attribute")
                .mode()
                .ungroup()
            )
            values.is_bool()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [False, True]

    def test_is_datetime(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"gender": "F", "datetime_attribute": datetime(2023, 1, 1)})
        )
        self.medrecord.add_nodes(
            ("pat_7", {"gender": "M", "datetime_attribute": datetime(2023, 1, 2)})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("datetime_attribute")
                .mode()
                .ungroup()
            )
            values.is_datetime()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
        ]

    def test_is_duration(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"gender": "F", "duration_attribute": timedelta(days=1)})
        )
        self.medrecord.add_nodes(
            ("pat_7", {"gender": "M", "duration_attribute": timedelta(days=2)})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("duration_attribute")
                .mode()
                .ungroup()
            )
            values.is_duration()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [
            timedelta(days=1),
            timedelta(days=2),
        ]

    def test_is_null(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"gender": "F", "null_attribute": None}))
        self.medrecord.add_nodes(("pat_7", {"gender": "M", "null_attribute": None}))

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("null_attribute")
                .mode()
                .ungroup()
            )
            values.is_null()
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [
            None,
            None,
        ]

    def test_is_max(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.is_max()
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.is_min()
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(32.66, rel=1e-2)]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.greater_than(40)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.greater_than_or_equal_to(59)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.less_than(50)
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(32.66, rel=1e-2)]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.less_than_or_equal_to(59)
            return values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [
            pytest.approx(32.66, rel=1e-2),
            59,
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.equal_to(59)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.not_equal_to(59)
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(32.66, rel=1e-2)]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.is_in([59])
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.is_not_in([59])
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(32.66, rel=1e-2)]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.starts_with(5)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.ends_with(9)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.add(1)
            return values

        assert self.medrecord.query_nodes(query) == [60]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.subtract(1)
            return values

        assert self.medrecord.query_nodes(query) == [58]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.multiply(2)
            return values

        assert self.medrecord.query_nodes(query) == [118]

    def test_divide(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.divide(2)
            return values

        assert self.medrecord.query_nodes(query) == [29.5]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.modulo(2)
            return values

        assert self.medrecord.query_nodes(query) == [1]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.power(2)
            return values

        assert self.medrecord.query_nodes(query) == [3481]

    def test_round(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.divide(2)
            values.round()
            return values

        assert self.medrecord.query_nodes(query) == [30]

    def test_ceil(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.divide(2)
            values.ceil()
            return values

        assert self.medrecord.query_nodes(query) == [30]

    def test_floor(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.divide(2)
            values.floor()
            return values

        assert self.medrecord.query_nodes(query) == [29]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.subtract(60)
            values.absolute()
            return values

        assert self.medrecord.query_nodes(query) == [1]

    def test_sqrt(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.contains(9)
            values.sqrt()
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(7.69, rel=1e-2)]

    def test_string_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            ("pat_6", {"string_attribute": " Hello World ", "gender": "M"})
        )

        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim()
            return values

        assert self.medrecord.query_nodes(query) == ["Hello World"]

        def query1(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim_start()
            return values

        assert self.medrecord.query_nodes(query1) == ["Hello World "]

        def query2(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim_end()
            return values

        assert self.medrecord.query_nodes(query2) == [" Hello World"]

        def query3(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.lowercase()
            return values

        assert self.medrecord.query_nodes(query3) == [" hello world "]

        def query4(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.uppercase()
            return values

        assert self.medrecord.query_nodes(query4) == [" HELLO WORLD "]

        def query5(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.slice(0, 6)
            return values

        assert self.medrecord.query_nodes(query5) == [" Hello"]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.either_or(
                lambda group: group.greater_than(50),
                lambda group: group.less_than(1),
            )
            return values

        assert self.medrecord.query_nodes(query) == [59]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            values.exclude(lambda group: group.greater_than(50))
            return values

        assert self.medrecord.query_nodes(query) == [pytest.approx(32.66, rel=1e-2)]

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithoutIndexOperand:
            values = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
            )
            clone_values = values.clone()
            values.greater_than(50)
            return clone_values

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x is None, x)
        ) == [
            pytest.approx(32.66, rel=1e-2),
            59,
        ]


class TestNodeMultipleValuesWithIndexOperand(unittest.TestCase):
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
        assert self.medrecord.query_nodes(
            lambda node: node.attribute("age").std()
        ) == pytest.approx(27.79, rel=1e-2)
        assert self.medrecord.query_nodes(
            lambda node: node.attribute("age").var()
        ) == pytest.approx(772.56, rel=1e-2)
        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").count()) == 5
        )
        assert (
            self.medrecord.query_nodes(lambda node: node.attribute("age").sum()) == 216
        )

        def query_random(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
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
        def query1(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.add(10)
            return age

        assert self.medrecord.query_nodes(query1) == {"pat_1": 52}

        def query2(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(10)
            return age

        assert self.medrecord.query_nodes(query2) == {"pat_1": 32}

        def query3(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.multiply(10)
            return age

        assert self.medrecord.query_nodes(query3) == {"pat_1": 420}

        def query4(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(10)
            return age

        assert self.medrecord.query_nodes(query4) == {"pat_1": 4.2}

        def query5(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.modulo(10)
            return age

        assert self.medrecord.query_nodes(query5) == {"pat_1": 2}

        def query6(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.power(2)
            return age

        assert self.medrecord.query_nodes(query6) == {"pat_1": 1764}

        def query7(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.floor()
            return age

        assert self.medrecord.query_nodes(query7) == {"pat_1": 8}

        def query8(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.ceil()
            return age

        assert self.medrecord.query_nodes(query8) == {"pat_1": 9}

        def query9(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.round()
            return age

        assert self.medrecord.query_nodes(query9) == {"pat_1": 8}

        def query10(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.sqrt()
            return age

        assert self.medrecord.query_nodes(query10) == {
            "pat_1": pytest.approx(6.48, rel=1e-2)
        }

        def query11(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(45)
            age.absolute()
            return age

        assert self.medrecord.query_nodes(query11) == {"pat_1": 3}

        def query12(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            age = node.attribute("gender")
            age.lowercase()
            return age

        assert self.medrecord.query_nodes(query12) == {"pat_1": "m"}

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"spacing": " hello "}))

        def query13(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            age = node.attribute("spacing")
            age.uppercase()
            return age

        assert self.medrecord.query_nodes(query13) == {"pat_6": " HELLO "}

        def query14(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            age = node.attribute("spacing")
            age.trim()
            return age

        assert self.medrecord.query_nodes(query14) == {"pat_6": "hello"}

        def query15(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            age = node.attribute("spacing")
            age.trim_start()
            return age

        assert self.medrecord.query_nodes(query15) == {"pat_6": "hello "}

        def query16(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            age = node.attribute("spacing")
            age.trim_end()
            return age

        assert self.medrecord.query_nodes(query16) == {"pat_6": " hello"}

        def query17(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            age = node.attribute("spacing")
            age.slice(0, 3)
            return age

        assert self.medrecord.query_nodes(query17) == {"pat_6": " he"}

        def query18(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            node.attribute("age").either_or(
                lambda attribute: attribute.greater_than(90),
                lambda attribute: attribute.less_than(20),
            )
            return node.attribute("age")

        assert self.medrecord.query_nodes(query18) == {"pat_3": 96, "pat_4": 19}

        def query19(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            node.attribute("age").exclude(
                lambda attribute: attribute.less_than(90),
            )
            return node.attribute("age")

        assert self.medrecord.query_nodes(query19) == {"pat_3": 96}

        def query20(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            query_node(node)
            clone = node.attribute("age").clone()
            node.attribute("age").add(10)
            return clone

        assert self.medrecord.query_nodes(query20) == {"pat_1": 42}


class TestEdgeMultipleValuesWithIndexOperand(unittest.TestCase):
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
        def query_min(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            query_specific_edge(edge, 4)
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.min()

        assert self.medrecord.query_edges(query_min) == (4, 0)

        def query_max(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.max()

        assert self.medrecord.query_edges(query_max) == (47, 3416)

        def query_mean(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.mean()

        assert self.medrecord.query_edges(query_mean) == pytest.approx(405.07, rel=1e-2)

        def query_median(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.median()

        assert self.medrecord.query_edges(query_median) == 315

        def query_mode(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.mode()

        assert self.medrecord.query_edges(query_mode) == 371

        def query_std(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.std()

        assert self.medrecord.query_edges(query_std) == pytest.approx(605.22, rel=1e-2)

        def query_var(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.var()

        assert self.medrecord.query_edges(query_var) == pytest.approx(
            366285.42, rel=1e-2
        )

        def query_count(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.count()

        assert self.medrecord.query_edges(query_count) == 43

        def query_sum(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            attribute = edge.attribute("duration_days")
            attribute.is_float()
            return attribute.sum()

        assert self.medrecord.query_edges(query_sum) == 17416

        def query_random(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
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
        def query1(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.add(10)
            return duration

        assert self.medrecord.query_edges(query1) == {3: 22}

        def query2(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.subtract(1)
            return duration

        assert self.medrecord.query_edges(query2) == {3: 11}

        def query3(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.multiply(5)
            return duration

        assert self.medrecord.query_edges(query3) == {3: 60}

        def query4(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            return duration

        assert self.medrecord.query_edges(query4) == {3: 3}

        def query5(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.modulo(3)
            return duration

        assert self.medrecord.query_edges(query5) == {3: 0}

        def query6(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.power(3)
            return duration

        assert self.medrecord.query_edges(query6) == {3: 1728}

        def query7(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(5)
            duration.floor()
            return duration

        assert self.medrecord.query_edges(query7) == {3: 2}

        def query8(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            duration.ceil()
            return duration

        assert self.medrecord.query_edges(query8) == {3: 3}

        def query9(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.divide(4)
            duration.round()
            return duration

        assert self.medrecord.query_edges(query9) == {3: 3}

        def query10(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.power(0.5)
            return duration

        assert self.medrecord.query_edges(query10) == {3: pytest.approx(3.46, rel=1e-2)}

        def query11(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            duration = edge.attribute("duration_days")
            duration.subtract(15)
            duration.absolute()
            return duration

        assert self.medrecord.query_edges(query11) == {3: 3}

        def query12(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.lowercase()
            return label

        assert self.medrecord.query_edges(query12) == {160: " hello "}

        def query13(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.uppercase()
            return label

        assert self.medrecord.query_edges(query13) == {160: " HELLO "}

        def query14(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.trim()
            return label

        assert self.medrecord.query_edges(query14) == {160: "Hello"}

        def query15(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.trim_start()
            return label

        assert self.medrecord.query_edges(query15) == {160: "Hello "}

        def query16(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.trim_end()
            return label

        assert self.medrecord.query_edges(query16) == {160: " Hello"}

        def query17(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            label = edge.attribute("string_attribute")
            label.slice(1, 3)
            return label

        assert self.medrecord.query_edges(query17) == {160: "He"}

        def query17_time_add(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 0)
            time_attr = edge.attribute("time")
            time_attr.add(timedelta(days=5))
            return time_attr

        assert self.medrecord.query_edges(query17_time_add) == {
            0: datetime(2014, 2, 11)
        }

        def query18_time_subtract(
            edge: EdgeOperand,
        ) -> EdgeMultipleValuesWithIndexOperand:
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

        def query21(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            clone = edge.attribute("duration_days").clone()
            edge.attribute("duration_days").add(10)
            return clone

        assert self.medrecord.query_edges(query21) == {3: 12}

        def query22(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, 3)
            attribute = edge.attribute("duration_days")
            attribute.sqrt()
            return attribute

        assert self.medrecord.query_edges(query22) == {3: pytest.approx(3.46, rel=1e-2)}


class TestEdgeMultipleValuesWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            (
                "pat_1",
                "pat_2",
                {
                    "string_attribute": " Hello ",
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "bool_attribute": True,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                },
            )
        )

    def sort_tuple(
        self,
        tuple_to_sort: tuple[GroupKey, EdgeMultipleValuesWithIndexQueryResult],
    ) -> tuple[GroupKey, EdgeMultipleValuesWithIndexQueryResult]:
        """Sorts the dictionary in a result tuple for consistent comparison."""
        value, nodes_with_attributes = tuple_to_sort
        return (
            value,
            dict(sorted(nodes_with_attributes.items(), key=operator.itemgetter(1))),
        )

    def test_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )

        assert sorted(self.medrecord.query_edges(query)) == [
            ("pat_1", (3, 12.0)),
            ("pat_3", (26, 1113.0)),
        ]

    def test_min(self) -> None:
        """test_min"""

        def query_min(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .min()
            )

        result = sorted(self.medrecord.query_edges(query_min))
        assert result == [
            ("pat_1", (4, 0.0)),
            ("pat_3", (28, 371.0)),
        ]

    def test_mean(self) -> None:
        """test_mean"""

        def query_mean(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
            )

        result = sorted(self.medrecord.query_edges(query_mean))
        assert result == [
            ("pat_1", 6.0),
            ("pat_3", 742.0),
        ]

    def test_median(self) -> None:
        """test_median"""

        def query_median(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .median()
            )

        result = sorted(self.medrecord.query_edges(query_median))
        assert result == [
            ("pat_1", 6.0),
            ("pat_3", 742.0),
        ]

    def test_mode(self) -> None:
        """test_mode"""

        def query_mode(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [4, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mode()
            )

        result = sorted(self.medrecord.query_edges(query_mode))
        assert result == [
            ("pat_1", 0.0),
            ("pat_3", 371.0),
        ]

    def test_sum(self) -> None:
        """test_sum"""

        def query_sum(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .sum()
            )

        result = sorted(self.medrecord.query_edges(query_sum))
        assert result == [
            ("pat_1", 12.0),
            ("pat_3", 1484.0),
        ]

    def test_count(self) -> None:
        """test_count"""

        def query_count(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .count()
            )

        result = sorted(self.medrecord.query_edges(query_count))
        assert result == [
            ("pat_1", 2),
            ("pat_3", 2),
        ]

    def test_random(self) -> None:
        """test_random"""

        def query_random(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [3, 26])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .random()
            )

        result = sorted(self.medrecord.query_edges(query_random))
        assert result == [
            ("pat_1", (3, 12.0)),
            ("pat_3", (26, 1113.0)),
        ]

    def test_std(self) -> None:
        """test_std"""

        def query_std(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .std()
            )

        result = sorted(self.medrecord.query_edges(query_std))
        assert result == [
            ("pat_1", pytest.approx(6.0)),
            ("pat_3", pytest.approx(371.0)),
        ]

    def test_var(self) -> None:
        """test_var"""

        def query_var(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .var()
            )

        result = sorted(self.medrecord.query_edges(query_var))
        assert result == [
            ("pat_1", pytest.approx(36.0)),
            ("pat_3", pytest.approx(137641.0)),
        ]

    def test_is_string(self) -> None:
        """test_is_string"""

        def query_is_string(
            edge: EdgeOperand,
        ) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.is_string()
            return group

        result = sorted(self.medrecord.query_edges(query_is_string))
        assert result == [
            ("pat_1", {160: " Hello "}),
        ]

    def test_is_int(self) -> None:
        """test_is_int"""

        def query_is_int(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "integer_attribute"
            )
            group.is_int()
            return group

        result = sorted(self.medrecord.query_edges(query_is_int))
        assert result == [
            ("pat_1", {160: 5}),
        ]

    def test_is_max(self) -> None:
        """test_is_max"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.is_max()
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [
            ("pat_1", {3: 12.0}),
            ("pat_3", {26: 1113.0}),
        ]

    def test_is_min(self) -> None:
        """test_is_min"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.is_min()
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [
            ("pat_1", {4: 0.0}),
            ("pat_3", {28: 371.0}),
        ]

    def test_is_float(self) -> None:
        """test_is_float"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.is_float()
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", {4: 0.0, 3: 12.0}),
            ("pat_3", {28: 371.0, 26: 1113.0}),
        ]

    def test_is_datetime(self) -> None:
        """test_is_datetime"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "time"
            )
            group.is_datetime()
            return group

        result = sorted(
            (self.sort_tuple(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", {3: datetime(2014, 10, 18), 4: datetime(2015, 4, 14)}),
            ("pat_3", {26: datetime(1991, 3, 3), 28: datetime(1994, 3, 20)}),
        ]

    def test_is_bool(self) -> None:
        """test_is_bool"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "bool_attribute"
            )
            group.is_bool()
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [
            ("pat_1", {160: True}),
        ]

    def test_is_null(self) -> None:
        """test_is_null"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "null_attribute"
            )
            group.is_null()
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [
            ("pat_1", {160: None}),
        ]

    def test_is_duration(self) -> None:
        """test_is_duration"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_attribute"
            )
            group.is_duration()
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [
            ("pat_1", {160: timedelta(hours=2)}),
        ]

    def test_greater_than(self) -> None:
        """test_greater_than"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.greater_than(100)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {}), ("pat_3", {28: 371.0, 26: 1113.0})]

    def test_greater_than_or_equal_to(self) -> None:
        """test_greater_than_or_equal_to"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.greater_than_or_equal_to(371)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {}), ("pat_3", {28: 371, 26: 1113.0})]

    def test_less_than(self) -> None:
        """test_less_than"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.less_than(300)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {4: 0.0, 3: 12.0}), ("pat_3", {})]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.less_than_or_equal_to(371)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {4: 0.0, 3: 12.0}), ("pat_3", {28: 371.0})]

    def test_equal_to(self) -> None:
        """test_equal_to"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.equal_to(12.0)
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [("pat_1", {3: 12.0}), ("pat_3", {})]

    def test_not_equal_to(self) -> None:
        """test_not_equal_to"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.not_equal_to(12.0)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {4: 0.0}), ("pat_3", {28: 371.0, 26: 1113.0})]

    def test_is_in(self) -> None:
        """test_is_in"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.is_in([12.0, 371.0, 500.0])
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [("pat_1", {3: 12.0}), ("pat_3", {28: 371.0})]

    def test_is_not_in(self) -> None:
        """test_is_not_in"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.is_not_in([12.0, 371.0, 500.0])
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [("pat_1", {4: 0.0}), ("pat_3", {26: 1113.0})]

    def test_string_operations(self) -> None:
        """test_starts_with"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.starts_with(" He")
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [("pat_1", {160: " Hello "})]

        def query1(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.ends_with("o ")
            return group

        assert self.medrecord.query_edges(query1) == [("pat_1", {160: " Hello "})]

        def query2(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.contains("Hello")
            return group

        assert self.medrecord.query_edges(query2) == [("pat_1", {160: " Hello "})]

        def query3(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.lowercase()
            return group

        assert self.medrecord.query_edges(query3) == [("pat_1", {160: " hello "})]

        def query4(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.uppercase()
            return group

        assert self.medrecord.query_edges(query4) == [("pat_1", {160: " HELLO "})]

        def query5(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.trim()
            return group

        assert self.medrecord.query_edges(query5) == [("pat_1", {160: "Hello"})]

        def query6(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.trim_start()
            return group

        assert self.medrecord.query_edges(query6) == [("pat_1", {160: "Hello "})]

        def query7(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.trim_end()
            return group

        assert self.medrecord.query_edges(query7) == [("pat_1", {160: " Hello"})]

        def query8(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, 160)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "string_attribute"
            )
            group.slice(1, 3)
            return group

        assert self.medrecord.query_edges(query8) == [("pat_1", {160: "He"})]

    def test_add(self) -> None:
        """test_add"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.add(10)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 10.0, 3: 22.0}),
            ("pat_3", {28: 381.0, 26: 1123.0}),
        ]

    def test_subtract(self) -> None:
        """test_subtract"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.subtract(10)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: -10.0, 3: 2.0}),
            ("pat_3", {28: 361.0, 26: 1103.0}),
        ]

    def test_multiply(self) -> None:
        """test_multiply"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.multiply(2)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 24.0}),
            ("pat_3", {28: 742.0, 26: 2226.0}),
        ]

    def test_divide(self) -> None:
        """test_divide"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.divide(2)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 6.0}),
            ("pat_3", {28: 185.5, 26: 556.5}),
        ]

    def test_power(self) -> None:
        """test_power"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.power(2)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 144.0}),
            ("pat_3", {28: 137641.0, 26: 1238769.0}),
        ]

    def test_modulo(self) -> None:
        """test_modulo"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.modulo(3)
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {4: 0.0, 3: 0.0}), ("pat_3", {26: 0.0, 28: 2.0})]

    def test_absolute(self) -> None:
        """test_absolute"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.subtract(100)
            group.absolute()
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {3: 88, 4: 100}),
            ("pat_3", {28: 271.0, 26: 1013.0}),
        ]

    def test_round(self) -> None:
        """test_round"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.divide(2)
            group.round()
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 6.0}),
            ("pat_3", {28: 186.0, 26: 557.0}),
        ]

    def test_ceil(self) -> None:
        """test_ceiling"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.divide(2)
            group.ceil()
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 6.0}),
            ("pat_3", {28: 186.0, 26: 557.0}),
        ]

    def test_floor(self) -> None:
        """test_floor"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.divide(2)
            group.floor()
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 6.0}),
            ("pat_3", {28: 185.0, 26: 556.0}),
        ]

    def test_sqrt(self) -> None:
        """test_sqrt"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.sqrt()
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: pytest.approx(3.464, rel=1e-2)}),
            (
                "pat_3",
                {
                    28: pytest.approx(19.261, rel=1e-2),
                    26: pytest.approx(33.362, rel=1e-2),
                },
            ),
        ]

    def test_either_or(self) -> None:
        """test_either_or"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.either_or(lambda g: g.equal_to(12.0), lambda g: g.equal_to(1113.0))
            return group

        result = sorted(self.medrecord.query_edges(query))
        assert result == [("pat_1", {3: 12.0}), ("pat_3", {26: 1113.0})]

    def test_exclude(self) -> None:
        """test_exclude"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            group.exclude(lambda g: g.greater_than(100))
            return group

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [("pat_1", {4: 0.0, 3: 12.0}), ("pat_3", {})]

    def test_ungroup(self) -> None:
        """test_ungroup"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            return (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .ungroup()
            )

        result = self.medrecord.query_edges(query)
        assert result == {3: 12.0, 4: 0.0, 26: 1113.0, 28: 371.0}

    def test_clone(self) -> None:
        """test_clone"""

        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).attribute(
                "duration_days"
            )
            clone = group.clone()
            group.add(1000)  # This modification should not affect the clone
            return clone

        result = sorted([self.sort_tuple(t) for t in self.medrecord.query_edges(query)])
        assert result == [
            ("pat_1", {4: 0.0, 3: 12.0}),
            ("pat_3", {28: 371.0, 26: 1113.0}),
        ]


class TestEdgeMultipleValuesWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            (
                "pat_1",
                "pat_2",
                {
                    "string_attribute": " Hello ",
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "bool_attribute": True,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                },
            )
        )

    def test_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.max()

        assert self.medrecord.query_edges(query) == 1007.25

    def test_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.min()

        assert self.medrecord.query_edges(query) == 123.4

    def test_mean(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.mean()

        assert self.medrecord.query_edges(query) == pytest.approx(404.83)

    def test_median(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.median()

        assert self.medrecord.query_edges(query) == 323.2

    def test_mode(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            return values.mode()

        assert self.medrecord.query_edges(query) == " Hello "

    def test_sum(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.sum()

        assert self.medrecord.query_edges(query) == pytest.approx(2024.15)

    def test_count(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.count()

        assert self.medrecord.query_edges(query) == 5

    def test_std(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.std()

        assert self.medrecord.query_edges(query) == pytest.approx(318.918, rel=1e-2)

    def test_var(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.var()

        assert self.medrecord.query_edges(query) == pytest.approx(101531, rel=1e-3)

    def test_is_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.is_max()
            return values

        assert self.medrecord.query_edges(query) == [1007.25]

    def test_is_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.is_min()
            return values

        assert self.medrecord.query_edges(query) == [123.4]

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.is_string()
            return values

        assert self.medrecord.query_edges(query) == [" Hello "]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
                .ungroup()
            )
            values.is_int()
            return values

        assert self.medrecord.query_edges(query) == [5]

    def test_is_datetime(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            query_specific_edge(edge, [3, 4, 26, 28])
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("time")
                .mean()
                .ungroup()
            )
            values.is_datetime()
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [
            datetime(1992, 9, 9, 12, 0),
            datetime(2015, 1, 15, 0, 0),
        ]

    def test_is_float(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.is_float()
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 162.2, 323.2, 408.1, 1007.25]

    def test_is_bool(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("bool_attribute")
                .mode()
                .ungroup()
            )
            values.is_bool()
            return values

        assert self.medrecord.query_edges(query) == [True]

    def test_is_duration(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_attribute")
                .mode()
                .ungroup()
            )
            values.is_duration()
            return values

        assert self.medrecord.query_edges(query) == [timedelta(hours=2)]

    def test_is_null(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("null_attribute")
                .mode()
                .ungroup()
            )
            values.is_null()
            return values

        assert self.medrecord.query_edges(query) == [None]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.less_than(408.1)
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 162.2, 323.2]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.less_than_or_equal_to(408.1)
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 162.2, 323.2, 408.1]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.equal_to(408.1)
            return values

        assert self.medrecord.query_edges(query) == [408.1]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.greater_than_or_equal_to(408.1)
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [408.1, 1007.25]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.not_equal_to(162.2)
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 323.2, 408.1, 1007.25]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.is_in([123.4, 323.2, 999.9])
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 323.2]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.is_not_in([162.2, 408.1])
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 323.2, 1007.25]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.starts_with(" ")
            return values

        assert self.medrecord.query_edges(query) == [" Hello "]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.ends_with(" ")
            return values

        assert self.medrecord.query_edges(query) == [" Hello "]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.contains("Hello")
            return values

        assert self.medrecord.query_edges(query) == [" Hello "]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.modulo(100)
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        expected = sorted([23.4, 62.2, 23.2, 8.1, 7.25])
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e, rel=1e-2)

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.subtract(400)
            values.absolute()
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        expected = sorted([276.6, 237.8, 76.8, 8.1, 607.25])
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e, rel=1e-2)

    def test_sqrt(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.sqrt()
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        assert result[0] == pytest.approx(11.108, rel=1e-2)
        assert result[-1] == pytest.approx(31.737, rel=1e-2)

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim()
            return values

        assert self.medrecord.query_edges(query) == ["Hello"]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim_start()
            return values

        assert self.medrecord.query_edges(query) == ["Hello "]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.trim_end()
            return values

        assert self.medrecord.query_edges(query) == [" Hello"]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.lowercase()
            return values

        assert self.medrecord.query_edges(query) == [" hello "]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.uppercase()
            return values

        assert self.medrecord.query_edges(query) == [" HELLO "]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
                .ungroup()
            )
            values.slice(1, 4)
            return values

        assert self.medrecord.query_edges(query) == ["Hel"]

    def test_random(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            query_specific_edge(edge, [3, 4, 26, 28])
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            return values.random()

        assert self.medrecord.query_edges(query) in [6, 742]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.multiply(2)
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        expected = [246.8, 324.4, 646.4, 816.2, 2014.5]
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e)

    def test_divide(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.divide(2)
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        expected = [61.7, 81.1, 161.6, 204.05, 503.625]
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e, rel=1e-2)

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.power(2)
            return values

        result = sorted(self.medrecord.query_edges(query), key=lambda x: (x is None, x))
        expected = [15227.56, 26308.84, 104458.24, 166545.61, 1014552.5625]
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e, rel=1e-2)

    def test_round(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.round()
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123, 162, 323, 408, 1007]

    def test_ceil(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.ceil()
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [124, 163, 324, 409, 1008]

    def test_floor(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.floor()
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123, 162, 323, 408, 1007]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.either_or(lambda v: v.equal_to(123.4), lambda v: v.equal_to(1007.25))
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 1007.25]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            values.exclude(lambda v: v.greater_than(400))
            return values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 162.2, 323.2]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            values = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .mean()
                .ungroup()
            )
            clone_of_values = values.clone()
            values.add(100)
            return clone_of_values

        assert sorted(
            self.medrecord.query_edges(query), key=lambda x: (x is None, x)
        ) == [123.4, 162.2, 323.2, 408.1, 1007.25]


class TestNodeSingleValueWithIndexOperand(unittest.TestCase):
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
        def query1(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.greater_than(90)
            return maximum

        assert self.medrecord.query_nodes(query1) == ("pat_3", 96)

        def query2(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            minimum = node.attribute("age").min()
            minimum.less_than(20)
            return minimum

        assert self.medrecord.query_nodes(query2) == ("pat_4", 19)

        def query3(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.equal_to(96)
            return maximum

        assert self.medrecord.query_nodes(query3) == ("pat_3", 96)

        def query4(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.not_equal_to(42)
            return maximum

        assert self.medrecord.query_nodes(query4) == ("pat_3", 96)

        def query5(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.is_in([96, 19])
            return maximum

        assert self.medrecord.query_nodes(query5) == ("pat_3", 96)

        def query6(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.is_not_in([42, 19])
            return maximum

        assert self.medrecord.query_nodes(query6) == ("pat_3", 96)

        def query7(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            minimum = node.attribute("age").min()
            minimum.less_than_or_equal_to(42)
            return minimum

        assert self.medrecord.query_nodes(query7) == ("pat_4", 19)

        def query8(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.greater_than_or_equal_to(96)
            return maximum

        assert self.medrecord.query_nodes(query8) == ("pat_3", 96)

        def query9(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.starts_with(9)
            return maximum

        assert self.medrecord.query_nodes(query9) == ("pat_3", 96)

        def query10(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.ends_with(6)
            return maximum

        assert self.medrecord.query_nodes(query10) == ("pat_3", 96)

        def query11(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.contains(9)
            return maximum

        assert self.medrecord.query_nodes(query11) == ("pat_3", 96)

    def test_node_single_value_operand_operations(self) -> None:
        def query1(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.add(10)
            return maximum

        assert self.medrecord.query_nodes(query1) == ("pat_3", 106)

        def query2(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.subtract(10)
            return maximum

        assert self.medrecord.query_nodes(query2) == ("pat_3", 86)

        def query3(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.multiply(10)
            return maximum

        assert self.medrecord.query_nodes(query3) == ("pat_3", 960)

        def query4(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.divide(10)
            return maximum

        assert self.medrecord.query_nodes(query4) == ("pat_3", 9.6)

        def query5(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.modulo(10)
            return maximum

        assert self.medrecord.query_nodes(query5) == ("pat_3", 6)

        def query6(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.power(2)
            return maximum

        assert self.medrecord.query_nodes(query6) == ("pat_3", 9216)

        def query7(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.floor()
            return maximum

        assert self.medrecord.query_nodes(query7) == ("pat_3", 19)

        def query8(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.ceil()
            return maximum

        assert self.medrecord.query_nodes(query8) == ("pat_3", 20)

        def query9(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.divide(5)
            maximum.round()
            return maximum

        assert self.medrecord.query_nodes(query9) == ("pat_3", 19)

        def query10(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.sqrt()
            return maximum

        assert self.medrecord.query_nodes(query10) == (
            "pat_3",
            pytest.approx(9.8, rel=1e-2),
        )

        def query11(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.subtract(100)
            maximum.absolute()
            return maximum

        assert self.medrecord.query_nodes(query11) == ("pat_3", 4)

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"spacing": " Hello "}))

        def query12(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.lowercase()
            return maximum

        assert self.medrecord.query_nodes(query12) == ("pat_6", " hello ")

        def query13(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.uppercase()
            return maximum

        assert self.medrecord.query_nodes(query13) == ("pat_6", " HELLO ")

        def query14(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim()
            return maximum

        assert self.medrecord.query_nodes(query14) == ("pat_6", "Hello")

        def query15(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim_start()
            return maximum

        assert self.medrecord.query_nodes(query15) == ("pat_6", "Hello ")

        def query16(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.trim_end()
            return maximum

        assert self.medrecord.query_nodes(query16) == ("pat_6", " Hello")

        def query17(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("spacing").max()
            maximum.slice(0, 3)
            return maximum

        assert self.medrecord.query_nodes(query17) == ("pat_6", " He")

        def query18(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.either_or(
                lambda value: value.greater_than(90),
                lambda value: value.less_than(20),
            )
            return maximum

        assert self.medrecord.query_nodes(query18) == ("pat_3", 96)

        def query19(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            maximum.exclude(
                lambda value: value.less_than(90),
            )
            return maximum

        assert self.medrecord.query_nodes(query19) == ("pat_3", 96)

        def query20(node: NodeOperand) -> NodeSingleValueWithIndexOperand:
            maximum = node.attribute("age").max()
            clone = maximum.clone()
            maximum.add(10)
            return clone

        assert self.medrecord.query_nodes(query20) == ("pat_3", 96)


class TestNodeSingleValueWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            (
                "pat_6",
                {
                    "gender": "M",
                    "string_attribute": " Hello ",
                    "bool_attribute": True,
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                    "datetime_attribute": datetime(2023, 10, 1),
                },
            )
        )

    def sort_tuples(
        self, tuples: list[tuple[GroupKey, NodeSingleValueWithIndexQueryResult]]
    ) -> list[tuple[GroupKey, NodeSingleValueWithIndexQueryResult]]:
        return sorted(tuples, key=lambda x: (x[0] is None, x[0], x[1]))

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.is_int()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96)),
            ("M", ("pat_1", 42)),
        ]

    def test_is_bool(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("bool_attribute")
                .max()
            )
            group.is_bool()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", True)),
        ]

    def test_is_float(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("float_attribute")
                .max()
            )
            group.is_float()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", 50.5)),
        ]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.is_string()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", " Hello ")),
        ]

    def test_is_datetime(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("datetime_attribute")
                .max()
            )
            group.is_datetime()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", datetime(2023, 10, 1))),
        ]

    def test_is_duration(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("duration_attribute")
                .max()
            )
            group.is_duration()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", timedelta(hours=2))),
        ]

    def test_is_null(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("null_attribute")
                .max()
            )
            group.is_null()
            return group

        assert self.medrecord.query_nodes(query) == [
            ("M", ("pat_6", None)),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.greater_than(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", None),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.greater_than_or_equal_to(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", ("pat_1", 42)),
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.equal_to(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", ("pat_1", 42)),
        ]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.less_than(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", None),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.less_than_or_equal_to(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", ("pat_1", 42)),
        ]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.not_equal_to(42)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", None),
        ]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.is_in([42, 96])
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", ("pat_1", 42)),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.is_not_in([42])
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", None),
        ]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.starts_with(9)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", None),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.ends_with(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", ("pat_1", 42)),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.contains(6)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96)),
            ("M", None),
        ]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.add(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 106.0)),
            ("M", ("pat_1", 52)),
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.subtract(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 86.0)),
            ("M", ("pat_1", 32)),
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.multiply(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 192.0)),
            ("M", ("pat_1", 84)),
        ]

    def test_divide(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.divide(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 48.0)),
            ("M", ("pat_1", 21)),
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.modulo(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 6.0)),
            ("M", ("pat_1", 2)),
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.power(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 9216.0)),
            ("M", ("pat_1", 1764)),
        ]

    def test_round(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.divide(3)
            group.round()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 32)),
            ("M", ("pat_1", 14)),
        ]

    def test_ceil(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.divide(3)
            group.ceil()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 32)),
            ("M", ("pat_1", 14)),
        ]

    def test_floor(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.divide(3)
            group.floor()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 32)),
            ("M", ("pat_1", 14)),
        ]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.subtract(100)
            group.absolute()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 4.0)),
            ("M", ("pat_1", 58)),
        ]

    def test_sqrt(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            node.has_attribute("gender")
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.sqrt()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", pytest.approx(9.8, rel=1e-2))),
            ("M", ("pat_1", pytest.approx(6.48, rel=1e-2))),
        ]

    def test_string_operations(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.trim()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", ("pat_6", "Hello")),
        ]

        def query1(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.trim_start()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query1)) == [
            ("M", ("pat_6", "Hello ")),
        ]

        def query2(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.trim_end()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query2)) == [
            ("M", ("pat_6", " Hello")),
        ]

        def query3(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.lowercase()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query3)) == [
            ("M", ("pat_6", " hello ")),
        ]

        def query4(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.uppercase()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query4)) == [
            ("M", ("pat_6", " HELLO ")),
        ]

        def query5(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.slice(0, 3)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query5)) == [
            ("M", ("pat_6", " He")),
        ]

        def query6(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            group.either_or(
                lambda g: g.equal_to(" Hello "),
                lambda g: g.equal_to("hello"),
            )
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query6)) == [
            ("M", ("pat_6", " Hello ")),
        ]

        def query7(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .max()
            )
            clone = group.clone()
            group.add("test")
            return clone

        assert self.sort_tuples(self.medrecord.query_nodes(query7)) == [
            ("M", ("pat_6", " Hello ")),
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            group.exclude(lambda group: group.equal_to(42))
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", None),
        ]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            return (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
                .ungroup()
            )

        assert self.medrecord.query_nodes(query) == {"pat_3": 96, "pat_1": 42}

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .max()
            )
            clone = group.clone()
            group.add(10)
            return clone

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", ("pat_3", 96.0)),
            ("M", ("pat_1", 42)),
        ]


class TestNodeSingleValueWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            (
                "pat_6",
                {
                    "gender": "M",
                    "string_attribute": " Hello ",
                    "bool_attribute": True,
                    "integer_attribute": 5,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                    "datetime_attribute": datetime(2023, 10, 1),
                },
            )
        )

    def test_is_float(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.is_float()
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("integer_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.is_int()
            return single_value

        assert self.medrecord.query_nodes(query) == 5

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.is_string()
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_is_datetime(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("datetime_attribute")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.is_datetime()
            return single_value

        assert self.medrecord.query_nodes(query) == datetime(2023, 10, 1)

    def test_is_duration(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("duration_attribute")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.is_duration()
            return single_value

        assert self.medrecord.query_nodes(query) == timedelta(hours=2)

    def test_is_null(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("null_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.is_null()
            return single_value

        assert self.medrecord.query_nodes(query) is None

    def test_is_bool(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("bool_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.is_bool()
            return single_value

        assert self.medrecord.query_nodes(query) is True

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.greater_than(40)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.greater_than_or_equal_to(45.833)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.less_than(100)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.less_than_or_equal_to(46)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.equal_to(42)
            return single_value

        assert self.medrecord.query_nodes(query) is None

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.not_equal_to(0)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.is_in([" Hello ", "World"])
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.is_not_in([0, 1])
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.starts_with(" ")
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.ends_with(" ")
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.contains("ell")
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.add(10)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(55.833, rel=1e-2)

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.subtract(10)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(35.833, rel=1e-2)

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.multiply(2)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(91.666, rel=1e-2)

    def test_divide(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.divide(2)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(22.916, rel=1e-2)

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.modulo(10)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(5.833, rel=1e-2)

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.power(2)
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(2102.6, rel=1e-1)

    def test_round(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.round()
            return single_value

        assert self.medrecord.query_nodes(query) == 46

    def test_ceil(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.ceil()
            return single_value

        assert self.medrecord.query_nodes(query) == 46

    def test_floor(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.floor()
            return single_value

        assert self.medrecord.query_nodes(query) == 45

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.subtract(100)
            single_value.absolute()
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(54.167, rel=1e-2)

    def test_sqrt(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.sqrt()
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(6.77, rel=1e-2)

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.trim()
            return single_value

        assert self.medrecord.query_nodes(query) == "Hello"

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.trim_start()
            return single_value

        assert self.medrecord.query_nodes(query) == "Hello "

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.trim_end()
            return single_value

        assert self.medrecord.query_nodes(query) == " Hello"

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.lowercase()
            return single_value

        assert self.medrecord.query_nodes(query) == " hello "

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.uppercase()
            return single_value

        assert self.medrecord.query_nodes(query) == " HELLO "

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
                .ungroup()
                .mode()
            )
            single_value.slice(1, 4)
            return single_value

        assert self.medrecord.query_nodes(query) == "Hel"

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.either_or(
                lambda v: v.less_than(46),
                lambda v: v.equal_to(100),
            )
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            single_value.exclude(lambda v: v.less_than(40))
            return single_value

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexOperand:
            single_value = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
                .ungroup()
                .mean()
            )
            clone = single_value.clone()
            single_value.add(100)
            return clone

        assert self.medrecord.query_nodes(query) == pytest.approx(45.833, rel=1e-2)


class TestNodeSingleValueWithoutIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(
            (
                "pat_6",
                {
                    "gender": "M",
                    "string_attribute": " Hello ",
                    "bool_attribute": True,
                    "integer_attribute": 5,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                    "datetime_attribute": datetime(2023, 10, 1),
                },
            )
        )

    def sort_tuples(
        self, tuples: list[tuple[GroupKey, NodeSingleValueWithoutIndexQueryResult]]
    ) -> list[tuple[GroupKey, NodeSingleValueWithoutIndexQueryResult]]:
        return sorted(tuples, key=lambda x: (x[0] is None, x[0], x[1]))

    def test_is_float(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.is_float()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59.0),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("integer_attribute")
                .mode()
            )
            group.is_int()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", 5),
        ]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.is_string()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", " Hello "),
        ]

    def test_is_datetime(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("datetime_attribute")
                .mean()
            )
            group.is_datetime()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", datetime(2023, 10, 1)),
        ]

    def test_is_duration(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("duration_attribute")
                .mean()
            )
            group.is_duration()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", timedelta(hours=2)),
        ]

    def test_is_null(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("null_attribute")
                .mode()
            )
            group.is_null()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", None),
        ]

    def test_is_bool(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("bool_attribute")
                .mode()
            )
            group.is_bool()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", True),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.greater_than(40)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", None),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.greater_than_or_equal_to(32.66)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.less_than(40)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.less_than_or_equal_to(59)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.equal_to(59)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", None),
        ]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.not_equal_to(59)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.is_in([59, 32.66])
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", None),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.is_not_in([59])
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.starts_with(" ")
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", " Hello "),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.ends_with(" ")
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", " Hello "),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.contains("ell")
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", " Hello "),
        ]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.add(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 69),
            ("M", pytest.approx(42.66, rel=1e-2)),
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.subtract(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 49),
            ("M", pytest.approx(22.66, rel=1e-2)),
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.multiply(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 118),
            ("M", pytest.approx(65.33, rel=1e-2)),
        ]

    def test_divide(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.divide(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 29.5),
            ("M", pytest.approx(16.33, rel=1e-2)),
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.modulo(10)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 9),
            ("M", pytest.approx(2.66, rel=1e-2)),
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.power(2)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 3481),
            ("M", pytest.approx(1067.56, rel=1e-2)),
        ]

    def test_round(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.round()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", 33),
        ]

    def test_ceil(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.ceil()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", 33),
        ]

    def test_floor(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.floor()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", 32),
        ]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.subtract(50)
            group.absolute()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 9),
            ("M", pytest.approx(17.33, rel=1e-2)),
        ]

    def test_sqrt(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.sqrt()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", pytest.approx(7.681, rel=1e-2)),
            ("M", pytest.approx(5.713, rel=1e-2)),
        ]

    def test_string_operations(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.trim()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("M", "Hello"),
        ]

        def query1(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.trim_start()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query1)) == [
            ("M", "Hello "),
        ]

        def query2(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.trim_end()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query2)) == [
            ("M", " Hello"),
        ]

        def query3(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.lowercase()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query3)) == [
            ("M", " hello "),
        ]

        def query4(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.uppercase()
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query4)) == [
            ("M", " HELLO "),
        ]

        def query5(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("string_attribute")
                .mode()
            )
            group.slice(0, 3)
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query5)) == [
            ("M", " He"),
        ]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.either_or(lambda g: g.equal_to(59), lambda g: g.less_than(50))
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            group.exclude(lambda g: g.equal_to(59))
            return group

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", None),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleValueWithoutIndexGroupOperand:
            group = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attribute("age")
                .mean()
            )
            clone = group.clone()
            group.add(100)
            return clone

        assert self.sort_tuples(self.medrecord.query_nodes(query)) == [
            ("F", 59),
            ("M", pytest.approx(32.66, rel=1e-2)),
        ]


class TestEdgeSingleValueWithIndexOperand(unittest.TestCase):
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
        def query1(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.is_string()
            return value

        assert self.medrecord.query_edges(query1) == (160, " Hello ")

        def query2(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.is_int()
            return value

        assert self.medrecord.query_edges(query2) == (160, 5)

        def query3(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            query_edge(edge)
            value = edge.attribute("time").max()
            value.is_datetime()
            return value

        assert self.medrecord.query_edges(query3) == (0, datetime(2014, 2, 6, 0, 0))

        def query4(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("float_attribute").max()
            value.is_float()
            return value

        assert self.medrecord.query_edges(query4) == (160, 50.5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("duration_attribute").max()
            value.is_duration()
            return value

        assert self.medrecord.query_edges(query5) == (163, timedelta(hours=2))

        def query6(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("bool_attribute").max()
            value.is_bool()
            return value

        assert self.medrecord.query_edges(query6) == (161, True)

        def query7(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("null_attribute").min()
            value.is_null()
            return value

        assert self.medrecord.query_edges(query7) == (162, None)

        def query8(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").max()
            value.is_float()
            return value

        assert self.medrecord.query_edges(query8) == (47, 3416)

    def test_edge_single_value_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.greater_than(4)
            return value

        assert self.medrecord.query_edges(query1) == (160, 5)

        def query2(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.less_than(10)
            return value

        assert self.medrecord.query_edges(query2) == (160, 5)

        def query3(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.equal_to(5)
            return value

        assert self.medrecord.query_edges(query3) == (160, 5)

        def query4(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.not_equal_to(10)
            return value

        assert self.medrecord.query_edges(query4) == (160, 5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.is_in([1, 3, 5, 7])
            return value

        assert self.medrecord.query_edges(query5) == (160, 5)

        def query6(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.is_not_in([1, 3, 7, 9])
            return value

        assert self.medrecord.query_edges(query6) == (160, 5)

        def query7(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.less_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query7) == (160, 5)

        def query8(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.greater_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query8) == (160, 5)

        def query9(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.starts_with(" H")
            return value

        assert self.medrecord.query_edges(query9) == (160, " Hello ")

        def query10(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.ends_with("o ")
            return value

        assert self.medrecord.query_edges(query10) == (160, " Hello ")

        def query11(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.contains("ell")
            return value

        assert self.medrecord.query_edges(query11) == (160, " Hello ")

    def test_edge_single_value_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.add(10)
            return value

        assert self.medrecord.query_edges(query1) == (160, 15)

        def query2(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.subtract(2)
            return value

        assert self.medrecord.query_edges(query2) == (160, 3)

        def query3(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.multiply(3)
            return value

        assert self.medrecord.query_edges(query3) == (160, 15)

        def query4(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.divide(2)
            return value

        assert self.medrecord.query_edges(query4) == (160, 2.5)

        def query5(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.modulo(3)
            return value

        assert self.medrecord.query_edges(query5) == (160, 2)

        def query6(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.power(2)
            return value

        assert self.medrecord.query_edges(query6) == (160, 25)

        def query7(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("float_attribute").max()
            value.floor()
            return value

        assert self.medrecord.query_edges(query7) == (160, 50.0)

        def query8(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("float_attribute").max()
            value.ceil()
            return value

        assert self.medrecord.query_edges(query8) == (160, 51.0)

        def query9(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("float_attribute").max()
            value.round()
            return value

        assert self.medrecord.query_edges(query9) == (160, 51.0)

        def query10(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.power(0.5)
            return value

        assert self.medrecord.query_edges(query10) == (
            160,
            pytest.approx(2.24, rel=1e-2),
        )

        def query11(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.subtract(10)
            value.absolute()
            return value

        assert self.medrecord.query_edges(query11) == (160, 5)

        def query12(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.lowercase()
            return value

        assert self.medrecord.query_edges(query12) == (160, " hello ")

        def query13(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.uppercase()
            return value

        assert self.medrecord.query_edges(query13) == (160, " HELLO ")

        def query14(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.trim()
            return value

        assert self.medrecord.query_edges(query14) == (160, "Hello")

        def query15(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.trim_start()
            return value

        assert self.medrecord.query_edges(query15) == (160, "Hello ")

        def query16(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()
            value.trim_end()
            return value

        assert self.medrecord.query_edges(query16) == (160, " Hello")

        def query17(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("string_attribute").max()  # " Hello "
            value.slice(1, 4)  # Slice "Hel"
            return value

        assert self.medrecord.query_edges(query17) == (160, "Hel")

        def query18(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            query_specific_edge(edge, 0)
            value = edge.attribute("time").max()
            value.add(timedelta(days=5))
            return value

        assert self.medrecord.query_edges(query18) == (0, datetime(2014, 2, 11))

        def query19(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            query_specific_edge(edge, 0)
            value = edge.attribute("time").max()
            value.subtract(timedelta(hours=1))
            return value

        assert self.medrecord.query_edges(query19) == (0, datetime(2014, 2, 5, 23, 0))

        def query20(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.either_or(
                lambda value: value.greater_than(3),
                lambda value: value.less_than(2),
            )
            return value

        assert self.medrecord.query_edges(query20) == (160, 5)

        def query21(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.exclude(
                lambda value: value.less_than(3),
            )
            return value

        assert self.medrecord.query_edges(query21) == (160, 5)

        def query22(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            original_val = edge.attribute("integer_attribute").max()
            cloned_val = original_val.clone()
            original_val.add(50)
            return cloned_val

        assert self.medrecord.query_edges(query22) == (160, 5)

        def query23(edge: EdgeOperand) -> EdgeSingleValueWithIndexOperand:
            value = edge.attribute("integer_attribute").max()
            value.sqrt()
            return value

        assert self.medrecord.query_edges(query23) == (
            160,
            pytest.approx(2.24, rel=1e-2),
        )


class TestEdgeSingleValueWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            (
                "pat_1",
                "pat_2",
                {
                    "string_attribute": " Hello ",
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "bool_attribute": True,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                },
            )
        )

    def sort_tuples(
        self, tuples: EdgeSingleValueWithIndexGroupQueryResult
    ) -> EdgeSingleValueWithIndexGroupQueryResult:
        return sorted(tuples, key=operator.itemgetter(0))

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.is_int()
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 5)),
        ]

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.is_string()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " Hello ")),
        ]

    def test_is_bool(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("bool_attribute").is_bool()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute("bool_attribute"))
                .attribute("bool_attribute")
                .max()
            )
            group.is_bool()
            return group

        assert self.medrecord.query_edges(query) == [
            (True, (160, True)),
        ]

    def test_is_null(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("null_attribute").is_null()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute("null_attribute"))
                .attribute("null_attribute")
                .max()
            )
            group.is_null()
            return group

        assert self.medrecord.query_edges(query) == [
            (None, (160, None)),
        ]

    def test_is_float(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("float_attribute").is_float()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("float_attribute")
                )
                .attribute("float_attribute")
                .max()
            )
            group.is_float()
            return group

        assert self.medrecord.query_edges(query) == [
            (50.5, (160, 50.5)),
        ]

    def test_is_duration(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("duration_attribute").is_duration()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("duration_attribute")
                )
                .attribute("duration_attribute")
                .max()
            )
            group.is_duration()
            return group

        assert self.medrecord.query_edges(query) == [
            (timedelta(hours=2), (160, timedelta(hours=2))),
        ]

    def test_is_datetime(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("time").is_datetime()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("time")
                .max()
            )
            group.is_datetime()
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", (115, datetime(2023, 5, 30, 13, 39, 26))),
            ("pat_2", (132, datetime(2024, 2, 20, 16, 58))),
            ("pat_3", (143, datetime(2001, 5, 20, 0, 38, 6))),
            ("pat_4", (150, datetime(2024, 4, 24, 3, 38, 35))),
            ("pat_5", (159, datetime(2024, 4, 12, 13, 23, 43))),
        ]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.greater_than(1000)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", (9, 1113)),
            ("pat_2", None),
            ("pat_3", (26, 1113)),
            ("pat_4", None),
            ("pat_5", (47, 3416)),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.greater_than_or_equal_to(1113)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", (9, 1113)),
            ("pat_2", None),
            ("pat_3", (26, 1113)),
            ("pat_4", None),
            ("pat_5", (47, 3416)),
        ]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.less_than(1000)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", None),
            ("pat_2", (19, 371)),
            ("pat_3", None),
            ("pat_4", (39, 371)),
            ("pat_5", None),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.less_than_or_equal_to(371)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", None),
            ("pat_2", (19, 371)),
            ("pat_3", None),
            ("pat_4", (39, 371)),
            ("pat_5", None),
        ]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.equal_to(371)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", None),
            ("pat_2", (19, 371)),
            ("pat_3", None),
            ("pat_4", (39, 371)),
            ("pat_5", None),
        ]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.not_equal_to(371)
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", (9, 1113)),
            ("pat_2", None),
            ("pat_3", (26, 1113)),
            ("pat_4", None),
            ("pat_5", (47, 3416)),
        ]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.is_in([371, 3416])
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", None),
            ("pat_2", (19, 371)),
            ("pat_3", None),
            ("pat_4", (39, 371)),
            ("pat_5", (47, 3416)),
        ]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            query_specific_edge(edge, [9, 19, 26, 39, 47])
            edge.attribute("duration_days").is_float()
            group = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_days")
                .max()
            )
            group.is_not_in([371])
            return group

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", (9, 1113)),
            ("pat_2", None),
            ("pat_3", (26, 1113)),
            ("pat_4", None),
            ("pat_5", (47, 3416)),
        ]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.starts_with(" ")
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " Hello ")),
        ]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.ends_with(" ")
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " Hello ")),
        ]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.contains("ell")
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " Hello ")),
        ]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.add(10)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 15)),
        ]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.subtract(2)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 3)),
        ]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.multiply(3)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 15)),
        ]

    def test_divide(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.divide(2)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 2.5)),
        ]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.modulo(3)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 2)),
        ]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.power(2)
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 25)),
        ]

    def test_round(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("float_attribute").is_float()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("float_attribute")
                )
                .attribute("float_attribute")
                .max()
            )
            group.round()
            return group

        assert self.medrecord.query_edges(query) == [
            (50.5, (160, 51.0)),
        ]

    def test_ceil(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("float_attribute").is_float()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("float_attribute")
                )
                .attribute("float_attribute")
                .max()
            )
            group.ceil()
            return group

        assert self.medrecord.query_edges(query) == [
            (50.5, (160, 51.0)),
        ]

    def test_floor(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("float_attribute").is_float()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("float_attribute")
                )
                .attribute("float_attribute")
                .max()
            )
            group.floor()
            return group

        assert self.medrecord.query_edges(query) == [
            (50.5, (160, 50.0)),
        ]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.subtract(10)
            group.absolute()
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 5)),
        ]

    def test_sqrt(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.sqrt()
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, pytest.approx(2.236, rel=1e-2))),
        ]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.trim()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, "Hello")),
        ]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.trim_start()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, "Hello ")),
        ]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.trim_end()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " Hello")),
        ]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.lowercase()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " hello ")),
        ]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.uppercase()
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, " HELLO ")),
        ]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("string_attribute").is_string()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("string_attribute")
                )
                .attribute("string_attribute")
                .max()
            )
            group.slice(1, 4)
            return group

        assert self.medrecord.query_edges(query) == [
            (" Hello ", (160, "Hel")),
        ]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.either_or(
                lambda g: g.equal_to(5),
                lambda g: g.equal_to(10),
            )
            return group

        assert self.medrecord.query_edges(query) == [
            (5, (160, 5)),
        ]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            group.exclude(lambda g: g.equal_to(5))
            return group

        assert self.medrecord.query_edges(query) == [
            (5, None),
        ]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            return group.ungroup()

        assert self.medrecord.query_edges(query) == {160: 5}

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithIndexGroupOperand:
            edge.attribute("integer_attribute").is_int()
            group = (
                edge.group_by(
                    EdgeOperandGroupDiscriminator.Attribute("integer_attribute")
                )
                .attribute("integer_attribute")
                .max()
            )
            clone = group.clone()
            group.add(10)
            return clone

        assert self.medrecord.query_edges(query) == [
            (5, (160, 5)),
        ]


class TestEdgeSingleValueWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            (
                "pat_1",
                "pat_2",
                {
                    "string_attribute": " Hello ",
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "bool_attribute": True,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                },
            )
        )

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.is_string()
            return value

        assert self.medrecord.query_edges(query) == " Hello "

    def test_is_float(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("float_attribute").mean()
            value.is_float()
            return value

        assert self.medrecord.query_edges(query) == 50.5

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("integer_attribute").is_int()
            value = edge.attribute("integer_attribute").mode()
            value.is_int()
            return value

        assert self.medrecord.query_edges(query) == 5

    def test_is_bool(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("bool_attribute").is_bool()
            value = edge.attribute("bool_attribute").mode()
            value.is_bool()
            return value

        assert self.medrecord.query_edges(query) is True

    def test_is_null(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("null_attribute").is_null()
            value = edge.attribute("null_attribute").mode()
            value.is_null()
            return value

        assert self.medrecord.query_edges(query) is None

    def test_is_duration(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_attribute").is_duration()
            value = edge.attribute("duration_attribute").mean()
            value.is_duration()
            return value

        assert self.medrecord.query_edges(query) == timedelta(hours=2)

    def test_is_datetime(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("time").is_datetime()
            value = edge.attribute("time").mean()
            value.is_datetime()
            return value

        assert self.medrecord.query_edges(query) == datetime(2014, 6, 7, 14, 4, 6)

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mean()
            value.greater_than(100)
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(405, rel=1e-2)

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mean()
            value.greater_than_or_equal_to(405)
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(405, rel=1e-2)

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mean()
            value.less_than(1000)
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(405, rel=1e-2)

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mode()
            value.less_than_or_equal_to(371)
            return value

        assert self.medrecord.query_edges(query) == 371

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.equal_to(" Hello ")
            return value

        assert self.medrecord.query_edges(query) == " Hello "

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mean()
            value.not_equal_to(0)
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(405, rel=1e-2)

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mode()
            value.is_in([371])
            return value

        assert self.medrecord.query_edges(query) == 371

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            edge.attribute("duration_days").is_float()
            value = edge.attribute("duration_days").mean()
            value.is_not_in([0, 1])
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(405, rel=1e-2)

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.starts_with(" ")
            return value

        assert self.medrecord.query_edges(query) == " Hello "

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.ends_with(" ")
            return value

        assert self.medrecord.query_edges(query) == " Hello "

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.contains("ell")
            return value

        assert self.medrecord.query_edges(query) == " Hello "

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.add(10)
            return value

        assert self.medrecord.query_edges(query) == 15

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.subtract(2)
            return value

        assert self.medrecord.query_edges(query) == 3

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.multiply(3)
            return value

        assert self.medrecord.query_edges(query) == 15

    def test_divide(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.divide(2)
            return value

        assert self.medrecord.query_edges(query) == 2.5

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.modulo(3)
            return value

        assert self.medrecord.query_edges(query) == 2

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.power(2)
            return value

        assert self.medrecord.query_edges(query) == 25

    def test_round(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("float_attribute").mean()
            value.round()
            return value

        assert self.medrecord.query_edges(query) == 51.0

    def test_ceil(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("float_attribute").mean()
            value.ceil()
            return value

        assert self.medrecord.query_edges(query) == 51.0

    def test_floor(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("float_attribute").mean()
            value.floor()
            return value

        assert self.medrecord.query_edges(query) == 50.0

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.subtract(10)
            value.absolute()
            return value

        assert self.medrecord.query_edges(query) == 5

    def test_sqrt(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.sqrt()
            return value

        assert self.medrecord.query_edges(query) == pytest.approx(2.236, rel=1e-2)

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.trim()
            return value

        assert self.medrecord.query_edges(query) == "Hello"

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.trim_start()
            return value

        assert self.medrecord.query_edges(query) == "Hello "

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.trim_end()
            return value

        assert self.medrecord.query_edges(query) == " Hello"

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.lowercase()
            return value

        assert self.medrecord.query_edges(query) == " hello "

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.uppercase()
            return value

        assert self.medrecord.query_edges(query) == " HELLO "

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("string_attribute").mode()
            value.slice(1, 4)
            return value

        assert self.medrecord.query_edges(query) == "Hel"

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.either_or(
                lambda v: v.greater_than(3),
                lambda v: v.less_than(2),
            )
            return value

        assert self.medrecord.query_edges(query) == 5

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            value.exclude(lambda v: v.less_than(3))
            return value

        assert self.medrecord.query_edges(query) == 5

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexOperand:
            value = edge.attribute("integer_attribute").mode()
            clone = value.clone()
            value.add(10)
            return clone

        assert self.medrecord.query_edges(query) == 5


class TestEdgeSingleValueWithoutIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            (
                "pat_1",
                "pat_2",
                {
                    "string_attribute": " Hello ",
                    "float_attribute": 50.5,
                    "integer_attribute": 5,
                    "bool_attribute": True,
                    "null_attribute": None,
                    "duration_attribute": timedelta(hours=2),
                },
            )
        )

    def sort_tuples(
        self, tuples: EdgeSingleValueWithoutIndexGroupQueryResult
    ) -> EdgeSingleValueWithoutIndexGroupQueryResult:
        return sorted(tuples, key=operator.itemgetter(0))

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.is_int()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.is_string()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " Hello "),
        ]

    def test_is_bool(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("bool_attribute")
                .mode()
            )
            value.is_bool()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", True),
        ]

    def test_is_null(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("null_attribute")
                .mode()
            )
            value.is_null()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", None),
        ]

    def test_is_duration(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("duration_attribute")
                .mean()
            )
            value.is_duration()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", timedelta(hours=2)),
        ]

    def test_is_datetime(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("time")
                .mean()
            )
            value.is_datetime()
            return value

        assert self.sort_tuples(self.medrecord.query_edges(query)) == [
            ("pat_1", datetime(2018, 7, 5, 22, 40, 30)),
            ("pat_2", datetime(2019, 12, 29, 22, 11, 7)),
            ("pat_3", datetime(1995, 11, 14, 21, 38, 52)),
            ("pat_4", datetime(2020, 11, 1, 14, 3, 11)),
            ("pat_5", datetime(2019, 2, 11, 6, 53, 55)),
        ]

    def test_is_float(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("float_attribute")
                .mean()
            )
            value.is_float()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 50.5),
        ]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.greater_than(4)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.greater_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.less_than(10)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.less_than_or_equal_to(5)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.equal_to(5)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.not_equal_to(10)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.is_in([1, 3, 5, 7])
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.is_not_in([1, 3, 7, 9])
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.starts_with(" ")
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " Hello "),
        ]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.ends_with(" ")
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " Hello "),
        ]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.contains("ell")
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " Hello "),
        ]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.add(10)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 15),
        ]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.subtract(2)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 3),
        ]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.multiply(3)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 15),
        ]

    def test_divide(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.divide(2)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 2.5),
        ]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.modulo(3)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 2),
        ]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.power(2)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 25),
        ]

    def test_round(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("float_attribute")
                .mean()
            )
            value.round()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 51.0),
        ]

    def test_ceil(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("float_attribute")
                .mean()
            )
            value.ceil()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 51.0),
        ]

    def test_floor(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("float_attribute")
                .mean()
            )
            value.floor()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 50.0),
        ]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.subtract(10)
            value.absolute()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_sqrt(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.sqrt()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", pytest.approx(2.236, rel=1e-2)),
        ]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.trim()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", "Hello"),
        ]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.trim_start()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", "Hello "),
        ]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.trim_end()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " Hello"),
        ]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.lowercase()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " hello "),
        ]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.uppercase()
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", " HELLO "),
        ]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("string_attribute")
                .mode()
            )
            value.slice(1, 4)
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", "Hel"),
        ]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.either_or(
                lambda v: v.greater_than(3),
                lambda v: v.less_than(2),
            )
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            value.exclude(lambda v: v.less_than(3))
            return value

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleValueWithoutIndexGroupOperand:
            value = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
                .attribute("integer_attribute")
                .mode()
            )
            clone = value.clone()
            value.add(10)
            return clone

        assert self.medrecord.query_edges(query) == [
            ("pat_1", 5),
        ]


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


class TestNodeIndicesGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes([(2, {"gender": "F"}), (3, {"gender": "M"})])

    def sort_tuples(
        self,
        tuple_to_sort: tuple[GroupKey, NodeIndicesQueryResult],
    ) -> tuple[GroupKey, NodeIndicesQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        return (value, sorted(nodes_with_attributes))

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.has_attribute("gender")
            node.index().is_string()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.max()

        assert sorted(
            self.medrecord.query_nodes(query),
            key=operator.itemgetter(0),
        ) == [
            ("F", "pat_3"),
            ("M", "pat_5"),
        ]

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.has_attribute("gender")
            node.index().is_string()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.min()

        assert sorted(
            self.medrecord.query_nodes(query),
            key=operator.itemgetter(0),
        ) == [
            ("F", "pat_2"),
            ("M", "pat_1"),
        ]

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.has_attribute("gender")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.count()

        assert sorted(
            self.medrecord.query_nodes(query),
            key=operator.itemgetter(0),
        ) == [
            ("F", 3),
            ("M", 4),
        ]

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.sum()

        assert sorted(
            self.medrecord.query_nodes(query),
            key=operator.itemgetter(0),
        ) == [
            ("F", 2),
            ("M", 3),
        ]

    def test_random(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.random()

        assert sorted(
            self.medrecord.query_nodes(query),
            key=operator.itemgetter(0),
        ) == [
            ("F", 2),
            ("M", 3),
        ]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_string()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_int()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", [3]),
        ]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_in(["pat_1", "pat_2"])
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_in(["pat_1", "pat_2"])
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2"]),
            ("M", ["pat_1"]),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_not_in(["pat_1", "pat_2"])
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_not_in(["pat_1", "pat_2"])
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_3"]),
            ("M", ["pat_4", "pat_5"]),
        ]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_string()
            node.index().starts_with("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.starts_with("pat_")
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().ends_with("_1")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.ends_with("_1")
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("M", ["pat_1"]),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_string()
            node.index().contains("at")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.contains("at")
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_1", "pat_4", "pat_5"]),
        ]

    def test_is_max(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_max()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_3"]),
            ("M", ["pat_5"]),
        ]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.is_min()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2"]),
            ("M", ["pat_1"]),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.greater_than(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", []),
            ("M", [3]),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.greater_than_or_equal_to(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", [3]),
        ]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.less_than(3)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", []),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.less_than_or_equal_to(3)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", [3]),
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.equal_to(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", []),
        ]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.not_equal_to(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", []),
            ("M", [3]),
        ]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.either_or(
                lambda idx: idx.equal_to("pat_1"),
                lambda idx: idx.equal_to("pat_2"),
            )
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2"]),
            ("M", ["pat_1"]),
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.exclude(lambda idx: idx.equal_to("pat_1"))
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", ["pat_2", "pat_3"]),
            ("M", ["pat_4", "pat_5"]),
        ]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.index().contains("pat_")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            return indices.ungroup()

        assert sorted(self.medrecord.query_nodes(query)) == [
            "pat_1",
            "pat_2",
            "pat_3",
            "pat_4",
            "pat_5",
        ]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.add(10)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [12]),
            ("M", [13]),
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.subtract(1)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [1]),
            ("M", [2]),
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.multiply(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [4]),
            ("M", [6]),
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.modulo(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [0]),
            ("M", [1]),
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.power(2)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [4]),
            ("M", [9]),
        ]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.subtract(5)
            indices.absolute()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [3]),
            ("M", [2]),
        ]

    def test_trim(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.trim()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", ["Hello"]) in result

    def test_trim_start(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.trim_start()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", ["Hello "]) in result

    def test_trim_end(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.trim_end()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", [" Hello"]) in result

    def test_slice(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.slice(0, 3)
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", [" He"]) in result

    def test_lowercase(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.lowercase()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", [" hello "]) in result

    def test_uppercase(self) -> None:
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().contains(" ")
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            indices.uppercase()
            return indices

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert ("M", [" HELLO "]) in result

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesGroupOperand:
            node.index().is_int()
            indices = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).index()
            clone = indices.clone()
            indices.add(100)
            return clone

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("F", [2]),
            ("M", [3]),
        ]


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


class TestNodeIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((5, {"gender": "M"}))
        self.medrecord.add_nodes((" Hello ", {"gender": "M"}))

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.is_string()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", "pat_3"),
            ("M", "pat_5"),
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.is_int()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.greater_than(4)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.greater_than_or_equal_to(5)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.less_than(5)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", None),
        ]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.less_than_or_equal_to(5)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.equal_to(5)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.not_equal_to(5)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [("M", None)]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.is_in([5])
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.is_not_in([5])
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [("M", None)]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.starts_with("pat_")
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", "pat_3"),
            ("M", "pat_5"),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.ends_with("_5")
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", None),
            ("M", "pat_5"),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.contains("at_")
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", "pat_3"),
            ("M", "pat_5"),
        ]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.add(10)
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 15),
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.subtract(1)
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 4),
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.multiply(2)
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 10),
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.modulo(2)
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 1),
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.power(2)
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 25),
        ]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.subtract(10)
            index.absolute()
            return index

        assert self.medrecord.query_nodes(query) == [
            ("M", 5),
        ]

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.trim()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", "Hello"),
        ]

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.trim_start()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", "Hello "),
        ]

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.trim_end()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", " Hello"),
        ]

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.slice(0, 3)
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", " He"),
        ]

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.lowercase()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", " hello "),
        ]

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains(" ")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.uppercase()
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", " HELLO "),
        ]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.either_or(
                lambda idx: idx.equal_to("pat_5"),
                lambda idx: idx.equal_to("pat_3"),
            )
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", "pat_3"),
            ("M", "pat_5"),
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            index.exclude(lambda idx: idx.equal_to("pat_5"))
            return index

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("F", "pat_3"),
            ("M", None),
        ]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.index().contains("pat_")
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            return index.ungroup()

        assert sorted(self.medrecord.query_nodes(query)) == [
            "pat_3",
            "pat_5",
        ]

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeIndexGroupOperand:
            node.index().is_int()
            index = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .index()
                .max()
            )
            clone = index.clone()
            index.add(100)
            return clone

        assert sorted(
            self.medrecord.query_nodes(query), key=lambda x: (x[0] is None, x[0])
        ) == [
            ("M", 5),
        ]


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


class TestEdgeIndicesGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.max()

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.min()

        assert self.medrecord.query_edges(query) == [("pat_1", 0)]

    def test_count(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.count()

        assert self.medrecord.query_edges(query) == [("pat_1", 10)]

    def test_sum(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.sum()

        assert self.medrecord.query_edges(query) == [("pat_1", 45)]

    def test_random(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().equal_to(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.random()

        assert self.medrecord.query_edges(query) == [("pat_1", 10)]

    def test_is_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.is_max()
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [9])]

    def test_is_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.is_min()
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [0])]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().is_in([0, 1, 2])
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.is_in([0, 1])
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1])]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().is_in([0, 1, 2])
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.is_not_in([0])
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [1, 2])]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.greater_than(5)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [6, 7, 8, 9])]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.greater_than_or_equal_to(5)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [5, 6, 7, 8, 9])]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.less_than(5)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1, 2, 3, 4])]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.less_than_or_equal_to(5)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1, 2, 3, 4, 5])]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.equal_to(3)
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [3])]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.not_equal_to(3)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1, 2, 4, 5, 6, 7, 8, 9])]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.either_or(
                lambda idx: idx.equal_to(1),
                lambda idx: idx.equal_to(2),
            )
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [1, 2])]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.exclude(lambda idx: idx.equal_to(1))
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 2, 3, 4, 5, 6, 7, 8, 9])]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            return indices.ungroup()

        assert sorted(self.medrecord.query_edges(query)) == list(range(10))

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.add(10)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]

    # TODO(@JabobKrauskopf, #386): underflow on negative edge indices
    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.subtract(1)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 4294967295])]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.multiply(2)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.modulo(2)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", sorted([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.power(2)
            return indices

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.starts_with(0)
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [0])]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.ends_with(0)
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [0])]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            indices.contains(5)
            return indices

        assert self.medrecord.query_edges(query) == [("pat_1", [5])]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesGroupOperand:
            edge.index().less_than(10)
            indices = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index()
            clone = indices.clone()
            indices.add(100)
            return clone

        result = [(k, sorted(v)) for k, v in self.medrecord.query_edges(query)]
        assert result == [("pat_1", list(range(10)))]


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


class TestEdgeIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.greater_than(5)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.greater_than_or_equal_to(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.less_than(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", None)]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.less_than_or_equal_to(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.equal_to(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.not_equal_to(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", None)]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.is_in([9])
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.is_not_in([9])
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", None)]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.starts_with(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.ends_with(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.contains(9)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.add(1)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 10)]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.subtract(1)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 8)]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.multiply(2)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 18)]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.modulo(2)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 1)]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.power(2)
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 81)]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.either_or(
                lambda idx: idx.equal_to(9),
                lambda idx: idx.equal_to(8),
            )
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            index.exclude(lambda idx: idx.equal_to(9))
            return index

        assert self.medrecord.query_edges(query) == [("pat_1", None)]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            return index.ungroup()

        assert self.medrecord.query_edges(query) == [9]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeIndexGroupOperand:
            edge.index().less_than(10)
            index = (
                edge.group_by(EdgeOperandGroupDiscriminator.SourceNode()).index().max()
            )
            clone = index.clone()
            index.add(10)
            return clone

        assert self.medrecord.query_edges(query) == [("pat_1", 9)]


class TestNodeAttributesTreeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_attributes_tree_operand_comparisons(self) -> None:
        def query1(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            return node.attributes().max()

        assert self.medrecord.query_nodes(query1) == {"pat_1": "gender"}

        def query2(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            return node.attributes().min()

        assert self.medrecord.query_nodes(query2) == {"pat_1": "age"}

        def query3(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            return node.attributes().count()

        assert self.medrecord.query_nodes(query3) == {"pat_1": 2}

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {1: "value1", 2: "value2"}))

        def query4(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(0)
            return node.attributes().sum()

        assert self.medrecord.query_nodes(query4) == {0: 3}

        def query5(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
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


class TestNodeAttributesTreeGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {10: "value1", 11: "value2"}))

    def sort_tuples_tree(
        self,
        tuple_to_sort: tuple[GroupKey, NodeAttributesTreeQueryResult],
    ) -> tuple[GroupKey, NodeAttributesTreeQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, sorted(v)) for k, v in nodes_with_attributes.items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def sort_tuples(
        self, tuple_to_sort: tuple[GroupKey, NodeMultipleAttributesWithIndexQueryResult]
    ) -> tuple[GroupKey, NodeMultipleAttributesWithIndexQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, v) for k, v in nodes_with_attributes.items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute("gender")
            return (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
            )

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": "gender", "pat_3": "gender"}),
            ("M", {"pat_1": "gender", "pat_4": "gender", "pat_5": "gender"}),
        ]

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute("gender")
            return (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .min()
            )

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": "age", "pat_3": "age"}),
            ("M", {"pat_1": "age", "pat_4": "age", "pat_5": "age"}),
        ]

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute("gender")
            return (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .count()
            )

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [
            ("F", {"pat_2": 2, "pat_3": 2}),
            ("M", {"pat_1": 2, "pat_4": 2, "pat_5": 2}),
        ]

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_6")
            attributes = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            return attributes.sum()

        assert self.medrecord.query_nodes(query) == [("value1", {"pat_6": 21})]

    def test_random(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_6")
            attributes = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            return attributes.random()

        assert self.medrecord.query_nodes(query) in [
            [("value1", {"pat_6": 10})],
            [("value1", {"pat_6": 11})],
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attributes = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attributes.is_int()
            return attributes

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10, 11]})]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.is_string()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["age", "gender"], "pat_3": ["age", "gender"]}),
            (
                "M",
                {
                    "pat_1": ["age", "gender"],
                    "pat_4": ["age", "gender"],
                    "pat_5": ["age", "gender"],
                },
            ),
            (None, {"pat_6": []}),
        ]

    def test_is_max(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.has_attribute("gender")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.is_max()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["gender"], "pat_3": ["gender"]}),
            ("M", {"pat_1": ["gender"], "pat_4": ["gender"], "pat_5": ["gender"]}),
        ]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.has_attribute("gender")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.is_min()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["age"], "pat_3": ["age"]}),
            ("M", {"pat_1": ["age"], "pat_4": ["age"], "pat_5": ["age"]}),
        ]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.is_in(["age"])
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["age"], "pat_3": ["age"]}),
            ("M", {"pat_1": ["age"], "pat_4": ["age"], "pat_5": ["age"]}),
            (None, {"pat_6": []}),
        ]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.is_not_in(["age"])
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["gender"], "pat_3": ["gender"]}),
            ("M", {"pat_1": ["gender"], "pat_4": ["gender"], "pat_5": ["gender"]}),
            (None, {"pat_6": [10, 11]}),
        ]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.starts_with("a")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["age"], "pat_3": ["age"]}),
            ("M", {"pat_1": ["age"], "pat_4": ["age"], "pat_5": ["age"]}),
            (None, {"pat_6": []}),
        ]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.ends_with("e")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["age"], "pat_3": ["age"]}),
            ("M", {"pat_1": ["age"], "pat_4": ["age"], "pat_5": ["age"]}),
            (None, {"pat_6": []}),
        ]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().starts_with("pat_")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            attrs.contains("gen")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [
            ("F", {"pat_2": ["gender"], "pat_3": ["gender"]}),
            ("M", {"pat_1": ["gender"], "pat_4": ["gender"], "pat_5": ["gender"]}),
            (None, {"pat_6": []}),
        ]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.greater_than(1)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10, 11]})]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.greater_than_or_equal_to(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [11]})]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.less_than(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10]})]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.less_than_or_equal_to(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10, 11]})]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.equal_to(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10]})]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.not_equal_to(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [11]})]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.either_or(
                lambda attr: attr.equal_to(10),
                lambda attr: attr.equal_to(11),
            )
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10, 11]})]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.exclude(lambda attr: attr.equal_to(10))
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [11]})]

    def test_trim(self) -> None:
        self.medrecord.add_nodes(("pat_7", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_7")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.trim()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_7": ["Hello"]})]

    def test_trim_start(self) -> None:
        self.medrecord.add_nodes(("pat_8", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_8")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.trim_start()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_8": ["Hello "]})]

    def test_trim_end(self) -> None:
        self.medrecord.add_nodes(("pat_9", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_9")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.trim_end()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_9": [" Hello"]})]

    def test_slice(self) -> None:
        self.medrecord.add_nodes(("pat_10", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_10")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.slice(0, 3)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_10": [" He"]})]

    def test_lowercase(self) -> None:
        self.medrecord.add_nodes(("pat_11", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_11")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.lowercase()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_11": [" hello "]})]

    def test_uppercase(self) -> None:
        self.medrecord.add_nodes(("pat_12", {" Hello ": "value"}))

        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_12")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(" Hello ")
            ).attributes()
            attrs.uppercase()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("value", {"pat_12": [" HELLO "]})]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.subtract(9)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [1, 2]})]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.multiply(2)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [20, 22]})]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.modulo(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [0, 1]})]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.power(2)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [100, 121]})]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            attrs.subtract(11)
            attrs.absolute()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [0, 1]})]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeOperand:
            node.has_attribute("gender")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute("gender")
            ).attributes()
            return attrs.ungroup()

        result = dict(
            sorted(
                {
                    key: sorted(value)
                    for key, value in self.medrecord.query_nodes(query).items()
                }.items()
            )
        )
        assert result == {
            "pat_1": ["age", "gender"],
            "pat_2": ["age", "gender"],
            "pat_3": ["age", "gender"],
            "pat_4": ["age", "gender"],
            "pat_5": ["age", "gender"],
        }

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeAttributesTreeGroupOperand:
            node.index().equal_to("pat_6")
            attrs = node.group_by(
                NodeOperandGroupDiscriminator.Attribute(10)
            ).attributes()
            clone = attrs.clone()
            attrs.add(10)
            return clone

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {"pat_6": [10, 11]})]


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
        def query1(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().max()

        assert self.medrecord.query_edges(query1) == {164: "b_attribute"}

        def query2(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().min()

        assert self.medrecord.query_edges(query2) == {164: "a_attribute"}

        def query3(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 164)
            return edge.attributes().count()

        assert self.medrecord.query_edges(query3) == {164: 2}

        def query4(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 160)
            return edge.attributes().sum()

        assert self.medrecord.query_edges(query4) == {160: 3}

        def query5(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
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


class TestEdgeAttributesTreeGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(
            [
                (
                    "pat_1",
                    "pat_2",
                    {
                        10: "value1",
                        11: "value2",
                    },
                ),
                ("pat_1", "pat_2", {" Hello ": "value"}),
            ]
        )

    def sort_tuples_tree(
        self,
        tuple_to_sort: tuple[GroupKey, EdgeAttributesTreeQueryResult],
    ) -> tuple[GroupKey, EdgeAttributesTreeQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, sorted(v)) for k, v in dict(nodes_with_attributes).items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def sort_tuples(
        self,
        tuple_to_sort: tuple[GroupKey, EdgeMultipleAttributesWithIndexQueryResult],
    ) -> tuple[GroupKey, EdgeMultipleAttributesWithIndexQueryResult]:
        key, inner_dict = tuple_to_sort
        sorted_inner = dict(sorted(inner_dict.items()))
        return (key, sorted_inner)

    def test_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            return group.attributes().max()

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            ("pat_1", {0: "time", 1: "time", 2: "time", 3: "time", 4: "time"})
        ]

    def test_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            group.index().less_than(5)
            return group.attributes().min()

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [
            (
                "pat_1",
                {
                    0: "duration_days",
                    1: "duration_days",
                    2: "duration_days",
                    3: "duration_days",
                    4: "duration_days",
                },
            )
        ]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.subtract(9)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [1, 2]})]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.multiply(2)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [20, 22]})]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.modulo(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [0, 1]})]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.power(2)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [100, 121]})]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.subtract(11)
            attrs.absolute()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [0, 1]})]

    def test_sum(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            return attrs.sum()

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("pat_1", {160: 21})]

    def test_random(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            return attrs.random()

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result in [
            [("pat_1", {160: 10})],
            [("pat_1", {160: 11})],
        ]

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_string()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        # No string attributes, so should be empty lists
        assert result == [("pat_1", {160: []})]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_int()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10, 11]})]

    def test_is_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_max()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_is_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_min()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10]})]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_in([10])
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10]})]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.is_not_in([10])
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.greater_than(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.greater_than_or_equal_to(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.less_than(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10]})]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.less_than_or_equal_to(11)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10, 11]})]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.equal_to(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10]})]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.not_equal_to(10)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.trim()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: ["Hello"]})]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.trim_start()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: ["Hello "]})]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.trim_end()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" Hello"]})]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.slice(0, 3)
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" He"]})]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.lowercase()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" hello "]})]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.contains("ell")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" Hello "]})]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.starts_with(" H")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" Hello "]})]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.ends_with(" ")
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" Hello "]})]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            return attrs.ungroup()

        result = self.medrecord.query_edges(query)
        result = {key: sorted(value) for key, value in result.items()}
        assert result == {161: [" Hello "]}

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.index().equal_to(161)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.uppercase()
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {161: [" HELLO "]})]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.either_or(
                lambda attr: attr.equal_to(10),
                lambda attr: attr.equal_to(11),
            )
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10, 11]})]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            attrs.exclude(lambda attr: attr.equal_to(10))
            return attrs

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [11]})]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeAttributesTreeGroupOperand:
            edge.has_attribute(10)
            group = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())
            attrs = group.attributes()
            clone = attrs.clone()
            attrs.add(10)
            return clone

        result = sorted(
            (self.sort_tuples_tree(item) for item in self.medrecord.query_edges(query)),
            key=lambda x: (x[0] is None, x[0]),
        )
        assert result == [("pat_1", {160: [10, 11]})]


class TestNodeMultipleAttributesWithIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_multiple_attributes_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            return node.attributes().max().max()

        assert self.medrecord.query_nodes(query1) == (1, 13)

        def query2(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            return node.attributes().min().min()

        assert self.medrecord.query_nodes(query2) == (0, 10)

        def query3(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.index().is_int()
            return node.attributes().max().count()

        assert self.medrecord.query_nodes(query3) == 2

        def query4(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.index().is_int()
            return node.attributes().min().sum()

        assert self.medrecord.query_nodes(query4) == 22

        def query5(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attributes = node.attributes().max()
            attributes.equal_to(13)
            return attributes.random()

        assert self.medrecord.query_nodes(query5) == (1, 13)

        def query6(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query6) == {0: 11, 1: 13}

        def query7(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query7) == {"pat_1": "gender"}

        def query8(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_max()
            return attribute

        assert self.medrecord.query_nodes(query8) == {1: 13}

        def query9(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.is_min()
            return attribute

        assert self.medrecord.query_nodes(query9) == {0: 10}

        def query10(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.greater_than(12)
            return attribute

        assert self.medrecord.query_nodes(query10) == {1: 13}

        def query11(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_nodes(query11) == {0: 10, 1: 12}

        def query12(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_nodes(query12) == {0: 11}

        def query13(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query13) == {0: 10, 1: 12}

        def query14(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query14) == {0: 11}

        def query15(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query15) == {1: 13}

        def query16(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_in([11])
            return attribute

        assert self.medrecord.query_nodes(query16) == {0: 11}

        def query17(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_nodes(query17) == {1: 13}

        def query18(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            attribute = node.attributes().min()
            attribute.starts_with("a")
            return attribute

        assert self.medrecord.query_nodes(query18) == {"pat_1": "age"}

        def query19(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            attribute = node.attributes().min()
            attribute.ends_with("e")
            return attribute

        assert self.medrecord.query_nodes(query19) == {"pat_1": "age"}

        def query20(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max()
            attribute.contains("ge")
            return attribute

        assert self.medrecord.query_nodes(query20) == {"pat_1": "gender"}

    def test_node_multiple_attributes_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query1) == {0: 13, 1: 15}

        def query2(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_nodes(query2) == {1: 11, 0: 9}

        def query3(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query3) == {0: 22, 1: 26}

        def query4(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query4) == {0: 1, 1: 1}

        def query5(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query5) == {0: 121, 1: 169}

        def query6(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.subtract(12)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query6) == {0: 1, 1: 1}

        def query7(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.either_or(
                lambda attribute: attribute.equal_to(13),
                lambda attribute: attribute.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query7) == {1: 13}

        def query8(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            attribute.exclude(
                lambda attribute: attribute.contains(13),
            )
            return attribute

        assert self.medrecord.query_nodes(query8) == {0: 11}

        self.medrecord.add_nodes((2, {" Hello ": "value1"}))

        def query9(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query9) == {2: "Hello"}

        def query10(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query10) == {2: "Hello "}

        def query11(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query11) == {2: " Hello"}

        def query12(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query12) == {2: " He"}

        def query13(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query13) == {2: " hello "}

        def query14(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query14) == {2: " HELLO "}

        def query15(node: NodeOperand) -> NodeMultipleValuesWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max()
            return attribute.to_values()

        assert self.medrecord.query_nodes(query15) == {
            0: "value2",
            1: "value4",
            2: "value1",
        }

        def query16(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.index().is_in([0, 1])
            attribute = node.attributes().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_nodes(query16) == {0: 11, 1: 13}


class TestNodeMultipleAttributesWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes(("pat_7", {10: "value1", 12: "value4"}))
        self.medrecord.add_nodes(("pat_8", {" Hello ": "value"}))

    def sort_tuples(
        self, tuple_to_sort: tuple[GroupKey, NodeMultipleAttributesWithIndexQueryResult]
    ) -> tuple[GroupKey, NodeMultipleAttributesWithIndexQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, v) for k, v in nodes_with_attributes.items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.max()

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.min()

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_6", 11))]

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.count()

        assert self.medrecord.query_nodes(query) == [("value1", 2)]

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            return attrs.sum()

        assert self.medrecord.query_nodes(query) == [("value1", 20)]

    def test_random(self) -> None:
        def query5(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.equal_to(12)
            return attrs.random()

        assert self.medrecord.query_nodes(query5) == [("value1", ("pat_7", 12))]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.is_string()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": " Hello "})]

    def test_is_int(self) -> None:
        def query6(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_int()
            return attrs

        assert self.medrecord.query_nodes(query6) == [
            ("value1", {"pat_6": 11, "pat_7": 12})
        ]

    def test_is_max(self) -> None:
        def query7(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_max()
            return attrs

        assert self.medrecord.query_nodes(query7) == [("value1", {"pat_7": 12})]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.add(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 13, "pat_7": 14})
        ]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.subtract(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 9, "pat_7": 10})
        ]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.multiply(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 22, "pat_7": 24})
        ]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.modulo(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 1, "pat_7": 0})
        ]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.power(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 121, "pat_7": 144})
        ]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.subtract(12)
            attrs.absolute()
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 1, "pat_7": 0})
        ]

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": "Hello"})]

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim_start()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": "Hello "})]

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim_end()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": " Hello"})]

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.slice(0, 3)
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": " He"})]

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.lowercase()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": " hello "})]

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.index().equal_to("pat_8")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.uppercase()
            return attrs

        assert self.medrecord.query_nodes(query) == [("value", {"pat_8": " HELLO "})]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.is_min()
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_nodes(query)),
            key=operator.itemgetter(0),
        )

        assert result == [("value1", {"pat_6": 10, "pat_7": 10})]

    def test_greater_than(self) -> None:
        def query9(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.greater_than(11)
            return attrs

        assert self.medrecord.query_nodes(query9) == [("value1", {"pat_7": 12})]

    def test_greater_than_or_equal_to(self) -> None:
        def query10(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.greater_than_or_equal_to(10)
            return attrs

        assert self.medrecord.query_nodes(query10) == [
            ("value1", {"pat_6": 10, "pat_7": 10})
        ]

    def test_less_than(self) -> None:
        def query11(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.less_than(12)
            return attrs

        assert self.medrecord.query_nodes(query11) == [("value1", {"pat_6": 11})]

    def test_less_than_or_equal_to(self) -> None:
        def query12(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.less_than_or_equal_to(11)
            return attrs

        assert self.medrecord.query_nodes(query12) == [
            ("value1", {"pat_6": 10, "pat_7": 10})
        ]

    def test_equal_to(self) -> None:
        def query13(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.equal_to(11)
            return attrs

        assert self.medrecord.query_nodes(query13) == [("value1", {"pat_6": 11})]

    def test_not_equal_to(self) -> None:
        def query14(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.not_equal_to(11)
            return attrs

        assert self.medrecord.query_nodes(query14) == [("value1", {"pat_7": 12})]

    def test_is_in(self) -> None:
        def query15(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_in([11])
            return attrs

        assert self.medrecord.query_nodes(query15) == [("value1", {"pat_6": 11})]

    def test_is_not_in(self) -> None:
        def query16(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_not_in([11])
            return attrs

        assert self.medrecord.query_nodes(query16) == [("value1", {"pat_7": 12})]

    def test_starts_with(self) -> None:
        def query17(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.starts_with(1)
            return attrs

        assert self.medrecord.query_nodes(query17) == [
            ("value1", {"pat_6": 10, "pat_7": 10})
        ]

    def test_ends_with(self) -> None:
        def query18(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.ends_with(0)
            return attrs

        assert self.medrecord.query_nodes(query18) == [
            ("value1", {"pat_6": 10, "pat_7": 10})
        ]

    def test_contains(self) -> None:
        def query19(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.contains(1)
            return attrs

        assert self.medrecord.query_nodes(query19) == [
            ("value1", {"pat_6": 11, "pat_7": 12})
        ]

    def test_to_values(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleValuesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.to_values()

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": "value2", "pat_7": "value4"})
        ]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.ungroup()

        assert self.medrecord.query_nodes(query) == {"pat_6": 11, "pat_7": 12}

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.either_or(
                lambda attr: attr.equal_to(11),
                lambda attr: attr.equal_to(12),
            )
            return attrs

        assert self.medrecord.query_nodes(query) == [
            ("value1", {"pat_6": 11, "pat_7": 12})
        ]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.exclude(lambda attr: attr.equal_to(11))
            return attrs

        assert self.medrecord.query_nodes(query) == [("value1", {"pat_7": 12})]

    def test_clone(self) -> None:
        def query20(node: NodeOperand) -> NodeMultipleAttributesWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            clone = attrs.clone()
            attrs.add(10)
            return clone

        assert self.medrecord.query_nodes(query20) == [
            ("value1", {"pat_6": 11, "pat_7": 12})
        ]


class TestNodeMultipleAttributesWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {" Hello ": "value1"}))
        self.medrecord.add_nodes(("pat_7", {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes(("pat_8", {10: "value1", 12: "value4"}))

    def test_max(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            return attrs.max()

        assert self.medrecord.query_nodes(query) == 3

    def test_min(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            return attrs.min()

        assert self.medrecord.query_nodes(query) == 2

    def test_count(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            return attrs.count()

        assert self.medrecord.query_nodes(query) == 2

    def test_sum(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            return attrs.sum()

        assert self.medrecord.query_nodes(query) == 2

    def test_random(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            return attrs.random()

        assert self.medrecord.query_nodes(query) in [2, 3]

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_string()
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [
            "gendergender",
            "gendergendergender",
        ]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.is_int()
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]

    def test_is_max(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.is_max()
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_is_min(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.is_min()
            return attrs

        assert self.medrecord.query_nodes(query) == [2]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.greater_than(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.greater_than_or_equal_to(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.less_than(3)
            return attrs

        assert self.medrecord.query_nodes(query) == [2]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.less_than_or_equal_to(3)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.equal_to(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [2]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.not_equal_to(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.is_in([2, 3])
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.is_not_in([3])
            return attrs

        assert self.medrecord.query_nodes(query) == [2]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.starts_with(2)
            return attrs

        assert self.medrecord.query_nodes(query) == [2]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.ends_with(3)
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.contains(3)
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.add(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [4, 5]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.subtract(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [0, 1]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.multiply(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [4, 6]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.modulo(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [0, 1]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.power(2)
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [4, 9]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.subtract(3)
            attrs.absolute()
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [0, 1]

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim()
            return attrs

        assert self.medrecord.query_nodes(query) == ["Hello"]

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim_start()
            return attrs

        assert self.medrecord.query_nodes(query) == ["Hello "]

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim_end()
            return attrs

        assert self.medrecord.query_nodes(query) == [" Hello"]

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.slice(0, 3)
            return attrs

        assert self.medrecord.query_nodes(query) == [" He"]

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.lowercase()
            return attrs

        assert self.medrecord.query_nodes(query) == [" hello "]

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.uppercase()
            return attrs

        assert self.medrecord.query_nodes(query) == [" HELLO "]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.either_or(
                lambda attr: attr.equal_to(2),
                lambda attr: attr.equal_to(3),
            )
            return attrs

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            attrs.exclude(lambda attr: attr.equal_to(2))
            return attrs

        assert self.medrecord.query_nodes(query) == [3]

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithoutIndexOperand:
            node.has_attribute("gender")
            attrs = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
                .attributes()
                .max()
                .count()
                .ungroup()
            )
            clone = attrs.clone()
            attrs.add(2)
            return clone

        assert sorted(self.medrecord.query_nodes(query)) == [2, 3]


class TestEdgeMultipleAttributesWithIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()

        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_3", "pat_4", {12: "value3", 13: "value4"})])
        self.medrecord.add_edges([("pat_5", "pat_1", {" Hello ": "value5"})])

    def test_edge_multiple_attributes_operand_comparisons(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().max().max()

        assert self.medrecord.query_edges(query1) == (161, 13)

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().min().min()

        assert self.medrecord.query_edges(query2) == (160, 10)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().max().count()

        assert self.medrecord.query_edges(query3) == 2

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.index().is_in([160, 161])
            return edge.attributes().min().sum()

        assert self.medrecord.query_edges(query4) == 22

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attributes = edge.attributes().max()
            attributes.equal_to(13)
            return attributes.random()

        assert self.medrecord.query_edges(query5) == (161, 13)

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query6) == {160: 11, 161: 13}

        def query7(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query7) == {0: "time"}

        def query8(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_max()
            return attribute

        assert self.medrecord.query_edges(query8) == {161: 13}

        def query9(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.is_min()
            return attribute

        assert self.medrecord.query_edges(query9) == {160: 10}

        def query10(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.greater_than(12)
            return attribute

        assert self.medrecord.query_edges(query10) == {161: 13}

        def query11(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_edges(query11) == {160: 10, 161: 12}

        def query12(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_edges(query12) == {160: 11}

        def query13(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query13) == {160: 10, 161: 12}

        def query14(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query14) == {160: 11}

        def query15(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query15) == {161: 13}

        def query16(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_in([11, 12])
            return attribute

        assert self.medrecord.query_edges(query16) == {160: 11}

        def query17(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.is_not_in([11, 12])
            return attribute

        assert self.medrecord.query_edges(query17) == {161: 13}

        def query18(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().min()
            attribute.starts_with("dur")
            return attribute

        assert self.medrecord.query_edges(query18) == {0: "duration_days"}

        def query19(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().min()
            attribute.ends_with("ys")
            return attribute

        assert self.medrecord.query_edges(query19) == {0: "duration_days"}

        def query20(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max()
            attribute.contains("im")
            return attribute

        assert self.medrecord.query_edges(query20) == {0: "time"}

    def test_edge_multiple_attributes_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query1) == {160: 13, 161: 15}

        def query2(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_edges(query2) == {160: 9, 161: 11}

        def query3(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query3) == {160: 22, 161: 26}

        def query4(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query4) == {160: 1, 161: 1}

        def query5(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query5) == {160: 121, 161: 169}

        def query6(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.subtract(12)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query6) == {160: 1, 161: 1}

        def query7(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.either_or(
                lambda attr: attr.equal_to(13),
                lambda attr: attr.equal_to(10),
            )
            return attribute

        assert self.medrecord.query_edges(query7) == {161: 13}

        def query8(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            attribute.exclude(
                lambda attr: attr.equal_to(13),
            )
            return attribute

        assert self.medrecord.query_edges(query8) == {160: 11}

        def query9(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query9) == {162: "Hello"}

        def query10(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query10) == {162: "Hello "}

        def query11(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query11) == {162: " Hello"}

        def query12(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query12) == {162: " He"}

        def query13(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query13) == {162: " hello "}

        def query14(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query14) == {162: " HELLO "}

        def query15(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexOperand:
            edge.index().is_in([160, 161, 162])
            attribute = edge.attributes().max()
            return attribute.to_values()

        assert self.medrecord.query_edges(query15) == {
            160: "value2",
            161: "value4",
            162: "value5",
        }

        def query16(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_edges(query16) == {160: 11, 161: 13}


class TestEdgeMultipleAttributesWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()

        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 12: "value4"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {" Hello ": "value5"})])

    def sort_tuples(
        self, tuple_to_sort: tuple[GroupKey, EdgeMultipleAttributesWithIndexQueryResult]
    ) -> tuple[GroupKey, EdgeMultipleAttributesWithIndexQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, v) for k, v in nodes_with_attributes.items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def sort_values(
        self, tuple_to_sort: tuple[GroupKey, EdgeMultipleValuesWithIndexQueryResult]
    ) -> tuple[GroupKey, EdgeMultipleValuesWithIndexQueryResult]:
        value, nodes_with_attributes = tuple_to_sort
        sorted_nodes = sorted(
            ((k, v) for k, v in nodes_with_attributes.items()),
            key=operator.itemgetter(0),
        )
        return (value, dict(sorted_nodes))

    def test_max(self) -> None:
        def query(node: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attrs = (
                node.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.max()

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.min()

        assert self.medrecord.query_edges(query) == [("value1", (160, 11))]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.add(2)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 13, 161: 14})]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.subtract(2)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 9, 161: 10})]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.multiply(2)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 22, 161: 24})]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.modulo(2)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 1, 161: 0})]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.power(2)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 121, 161: 144})]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.subtract(12)
            attrs.absolute()
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )
        assert result == [("value1", {160: 1, 161: 0})]

    def test_count(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.count()

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_sum(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            return attrs.sum()

        assert self.medrecord.query_edges(query) == [("value1", 20)]

    def test_random(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.equal_to(11)
            return attrs.random()

        assert self.medrecord.query_edges(query) == [("value1", (160, 11))]

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.is_string()
            return attrs

        assert self.medrecord.query_edges(query) == [("value5", {162: " Hello "})]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_int()
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [("value1", {160: 11, 161: 12})]

    def test_is_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_max()
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {161: 12})]

    def test_is_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.is_min()
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 10, 161: 10})]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.greater_than(11)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {161: 12})]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.greater_than_or_equal_to(10)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 10, 161: 10})]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.less_than(13)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [("value1", {160: 11, 161: 12})]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.less_than_or_equal_to(11)
            return attrs

        result = sorted(
            (self.sort_tuples(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [("value1", {160: 10, 161: 10})]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.equal_to(11)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 11})]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.not_equal_to(11)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {161: 12})]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_in([11])
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 11})]

    def test_string_operations(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim()
            return attrs

        assert self.medrecord.query_edges(query) == [("value5", {162: "Hello"})]

        def query1(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim_start()
            return attrs

        assert self.medrecord.query_edges(query1) == [("value5", {162: "Hello "})]

        def query2(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.trim_end()
            return attrs

        assert self.medrecord.query_edges(query2) == [("value5", {162: " Hello"})]

        def query3(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.slice(0, 3)
            return attrs

        assert self.medrecord.query_edges(query3) == [("value5", {162: " He"})]

        def query4(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.lowercase()
            return attrs

        assert self.medrecord.query_edges(query4) == [("value5", {162: " hello "})]

        def query5(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
            )
            attrs.uppercase()
            return attrs

        assert self.medrecord.query_edges(query5) == [("value5", {162: " HELLO "})]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.is_not_in([11])
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {161: 12})]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.starts_with(1)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 10, 161: 10})]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .min()
            )
            attrs.ends_with(0)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 10, 161: 10})]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.contains(1)
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 11, 161: 12})]

    def test_to_values(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleValuesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.to_values()

        result = sorted(
            (self.sort_values(item) for item in self.medrecord.query_edges(query)),
            key=operator.itemgetter(0),
        )

        assert result == [("value1", {160: "value2", 161: "value4"})]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            return attrs.ungroup()

        assert self.medrecord.query_edges(query) == {160: 11, 161: 12}

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.either_or(
                lambda attr: attr.equal_to(11),
                lambda attr: attr.equal_to(12),
            )
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {160: 11, 161: 12})]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            attrs.exclude(lambda attr: attr.equal_to(11))
            return attrs

        assert self.medrecord.query_edges(query) == [("value1", {161: 12})]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexGroupOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
            )
            clone = attrs.clone()
            attrs.add(10)
            return clone

        assert self.medrecord.query_edges(query) == [("value1", {160: 11, 161: 12})]


class TestEdgeMultipleAttributesWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 12: "value4"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {" Hello ": "value5"})])

    def test_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            return attrs.max()

        assert self.medrecord.query_edges(query) == 23

    def test_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            return attrs.min()

        assert self.medrecord.query_edges(query) == 23

    def test_count(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            return attrs.count()

        assert self.medrecord.query_edges(query) == 1

    def test_sum(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            return attrs.sum()

        assert self.medrecord.query_edges(query) == 23

    def test_random(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            return attrs.random()

        assert self.medrecord.query_edges(query) == 23

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_string()
            return attrs

        assert self.medrecord.query_edges(query) == [" Hello "]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_int()
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [23]

    def test_is_max(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_max()
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_is_min(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_min()
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.greater_than(11)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.greater_than_or_equal_to(11)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [23]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.less_than(25)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.less_than_or_equal_to(23)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [23]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.equal_to(23)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.not_equal_to(22)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_in([23, 12])
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [23]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.is_not_in([12])
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.starts_with(2)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.ends_with(3)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.contains(2)
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.add(2)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [25]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.subtract(10)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [13]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.multiply(2)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [46]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.modulo(2)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [1]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.power(2)
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [529]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.subtract(25)
            attrs.absolute()
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [2]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim()
            return attrs

        assert self.medrecord.query_edges(query) == ["Hello"]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim_start()
            return attrs

        assert self.medrecord.query_edges(query) == ["Hello "]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.trim_end()
            return attrs

        assert self.medrecord.query_edges(query) == [" Hello"]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.slice(0, 3)
            return attrs

        assert self.medrecord.query_edges(query) == [" He"]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.lowercase()
            return attrs

        assert self.medrecord.query_edges(query) == [" hello "]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.uppercase()
            return attrs

        assert self.medrecord.query_edges(query) == [" HELLO "]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.either_or(
                lambda attr: attr.equal_to(23),
                lambda attr: attr.equal_to(12),
            )
            return attrs

        assert sorted(self.medrecord.query_edges(query)) == [23]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            attrs.exclude(lambda attr: attr.equal_to(11))
            return attrs

        assert self.medrecord.query_edges(query) == [23]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithoutIndexOperand:
            edge.has_attribute(10)
            attrs = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
                .ungroup()
            )
            clone = attrs.clone()
            attrs.add(2)
            return clone

        assert sorted(self.medrecord.query_edges(query)) == [23]


class TestNodeSingleAttributeWithIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_single_attribute_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query1) == ("pat_1", "gender")

        def query2(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query2) == (0, 10)

        def query3(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_nodes(query3) == (1, 13)

        def query4(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_nodes(query4) == (0, 10)

        def query5(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_nodes(query5) == (0, 10)

        def query6(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query6) == (0, 10)

        def query7(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.equal_to(13)
            return attribute

        assert self.medrecord.query_nodes(query7) == (1, 13)

        def query8(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query8) == (1, 13)

        def query9(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.is_in([13])
            return attribute

        assert self.medrecord.query_nodes(query9) == (1, 13)

        def query10(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_nodes(query10) == (1, 13)

        def query11(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.starts_with("g")
            return attribute

        assert self.medrecord.query_nodes(query11) == ("pat_1", "gender")

        def query12(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.ends_with("er")
            return attribute

        assert self.medrecord.query_nodes(query12) == ("pat_1", "gender")

        def query13(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            query_node(node)
            attribute = node.attributes().max().max()
            attribute.contains("ge")
            return attribute

        assert self.medrecord.query_nodes(query13) == ("pat_1", "gender")

        self.medrecord.add_nodes((2, {" Hello ": "value1"}))

        def query14(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query14) == (2, "Hello")

        def query15(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query15) == (2, "Hello ")

        def query16(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query16) == (2, " Hello")

        def query17(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query17) == (2, " He")

        def query18(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query18) == (2, " hello ")

        def query19(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().equal_to(2)
            attribute = node.attributes().max().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query19) == (2, " HELLO ")

    def test_single_attribute_operand_operations(self) -> None:
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes((1, {12: "value3", 13: "value4"}))

        def query1(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query1) == (1, 15)

        def query2(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_nodes(query2) == (1, 11)

        def query3(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().min().min()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query3) == (0, 20)

        def query4(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query4) == (1, 1)

        def query5(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query5) == (1, 169)

        def query6(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.subtract(14)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query6) == (1, 1)

        def query7(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.either_or(
                lambda attribute: attribute.equal_to(13),
                lambda attribute: attribute.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query7) == (1, 13)

        def query8(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            attribute.exclude(
                lambda attribute: attribute.contains(12),
            )
            return attribute

        assert self.medrecord.query_nodes(query8) == (1, 13)

        def query9(node: NodeOperand) -> NodeSingleAttributeWithIndexOperand:
            node.index().is_int()
            attribute = node.attributes().max().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_nodes(query9) == (1, 13)


class TestNodeSingleAttributeWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes(("pat_7", {10: "value1", 12: "value2"}))
        self.medrecord.add_nodes(("pat_8", {" Hello ": "value5"}))

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " Hello "))]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 14))]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 10))]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 24))]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 0))]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 144))]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.subtract(13)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 1))]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.greater_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.less_than(13)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.equal_to(12)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_in([12])
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " Hello "))]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " Hello "))]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " Hello "))]

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", "Hello"))]

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", "Hello "))]

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " Hello"))]

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " He"))]

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " hello "))]

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", ("pat_8", " HELLO "))]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.either_or(
                lambda attr: attr.equal_to(12),
                lambda attr: attr.equal_to(11),
            )
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.exclude(
                lambda attr: attr.equal_to(11),
            )
            return attribute

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]

    def test_ungroup(self) -> None:
        def query(node: NodeOperand) -> NodeMultipleAttributesWithIndexOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            return attribute.ungroup()

        assert self.medrecord.query_nodes(query) == {"pat_7": 12}

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            clone = attribute.clone()
            attribute.add(2)
            return clone

        assert self.medrecord.query_nodes(query) == [("value1", ("pat_7", 12))]


class TestNodeSingleAttributeWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes(("pat_7", {" Hello ": "value3"}))

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.greater_than(20)
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.greater_than_or_equal_to(21)
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.less_than(21)
            return attribute

        assert self.medrecord.query_nodes(query) is None

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.less_than_or_equal_to(21)
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.equal_to(21)
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.not_equal_to(22)
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.is_in([21, 22])
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.is_not_in([22])
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_nodes(query) == 23

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.subtract(1)
            return attribute

        assert self.medrecord.query_nodes(query) == 20

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_nodes(query) == 42

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_nodes(query) == 1

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_nodes(query) == 441

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.subtract(23)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_nodes(query) == 2

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_nodes(query) == " Hello "

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query) == "Hello"

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query) == "Hello "

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query) == " Hello"

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query) == " He"

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query) == " hello "

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(" Hello ")
            attribute = node.attributes().sum().sum()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query) == " HELLO "

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.either_or(
                lambda attr: attr.equal_to(21),
                lambda attr: attr.equal_to(22),
            )
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            attribute.exclude(lambda attr: attr.equal_to(22))
            return attribute

        assert self.medrecord.query_nodes(query) == 21

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexOperand:
            node.has_attribute(10)
            attribute = node.attributes().sum().sum()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_nodes(query) == 21


class TestNodeSingleAttributeWithoutIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {10: "value1", 11: "value2"}))
        self.medrecord.add_nodes(("pat_7", {10: "value3", 12: "value4"}))
        self.medrecord.add_nodes(("pat_8", {" Hello ": "value5"}))

    def test_is_string(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.is_string()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " Hello ")]

    def test_is_int(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.is_int()
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", 12)]

    def test_greater_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.greater_than(11)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", None), ("value3", 12)]

    def test_greater_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.greater_than_or_equal_to(12)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", None), ("value3", 12)]

    def test_less_than(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.less_than(12)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_less_than_or_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.less_than_or_equal_to(11)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.equal_to(11)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_not_equal_to(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.not_equal_to(12)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_is_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.is_in([11, 12])
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", 12)]

    def test_is_not_in(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.is_not_in([12])
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_add(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.add(2)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 13), ("value3", 14)]

    def test_subtract(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.subtract(2)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 9), ("value3", 10)]

    def test_multiply(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.multiply(2)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 22), ("value3", 24)]

    def test_modulo(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.modulo(2)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 1), ("value3", 0)]

    def test_power(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.power(2)
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 121), ("value3", 144)]

    def test_absolute(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.subtract(12)
            attribute.absolute()
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 1), ("value3", 0)]

    def test_starts_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " Hello ")]

    def test_ends_with(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " Hello ")]

    def test_contains(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " Hello ")]

    def test_trim(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", "Hello")]

    def test_trim_start(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", "Hello ")]

    def test_trim_end(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " Hello")]

    def test_slice(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " He")]

    def test_lowercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " hello ")]

    def test_uppercase(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(" Hello ")
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_nodes(query) == [("value5", " HELLO ")]

    def test_either_or(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.either_or(
                lambda attr: attr.equal_to(11),
                lambda attr: attr.equal_to(12),
            )
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", 12)]

    def test_exclude(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            attribute.exclude(lambda attr: attr.equal_to(12))
            return attribute

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", None)]

    def test_clone(self) -> None:
        def query(node: NodeOperand) -> NodeSingleAttributeWithoutIndexGroupOperand:
            node.has_attribute(10)
            attribute = (
                node.group_by(NodeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .sum()
            )
            clone = attribute.clone()
            attribute.add(2)
            return clone

        assert sorted(
            self.medrecord.query_nodes(query), key=operator.itemgetter(0)
        ) == [("value1", 11), ("value3", 12)]


class TestEdgeSingleAttributeWithIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_3", "pat_4", {12: "value3", 13: "value4"})])
        self.medrecord.add_edges([("pat_5", "pat_1", {" Hello ": "value5"})])

    def test_single_attribute_operand_comparisons(self) -> None:
        self.medrecord.unfreeze_schema()

        def query1(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query1) == (0, "time")

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query2) == (160, 10)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_edges(query3) == (161, 13)

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.greater_than_or_equal_to(10)
            return attribute

        assert self.medrecord.query_edges(query4) == (160, 10)

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.less_than(12)
            return attribute

        assert self.medrecord.query_edges(query5) == (160, 10)

        def query6(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query6) == (160, 10)

        def query7(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.equal_to(13)
            return attribute

        assert self.medrecord.query_edges(query7) == (161, 13)

        def query8(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query8) == (161, 13)

        def query9(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.is_in([13])
            return attribute

        assert self.medrecord.query_edges(query9) == (161, 13)

        def query10(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_edges(query10) == (161, 13)

        def query11(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.starts_with("t")
            return attribute

        assert self.medrecord.query_edges(query11) == (0, "time")

        def query12(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.ends_with("me")
            return attribute

        assert self.medrecord.query_edges(query12) == (0, "time")

        def query13(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 0)
            attribute = edge.attributes().max().max()
            attribute.contains("im")
            return attribute

        assert self.medrecord.query_edges(query13) == (0, "time")

        def query14(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query14) == (162, "Hello")

        def query15(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query15) == (162, "Hello ")

        def query16(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query16) == (162, " Hello")

        def query17(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query17) == (162, " He")

        def query18(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query18) == (162, " hello ")

        def query19(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            query_specific_edge(edge, 162)
            attribute = edge.attributes().max().max()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query19) == (162, " HELLO ")

    def test_single_attribute_operand_operations(self) -> None:
        def query1(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query1) == (161, 15)

        def query2(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_edges(query2) == (161, 11)

        def query3(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().min().min()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query3) == (160, 20)

        def query4(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query4) == (161, 1)

        def query5(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query5) == (161, 169)

        def query6(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.subtract(14)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query6) == (161, 1)

        def query7(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.either_or(
                lambda attr: attr.equal_to(13),
                lambda attr: attr.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_edges(query7) == (161, 13)

        def query8(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            attribute.exclude(
                lambda attr: attr.contains(12),
            )
            return attribute

        assert self.medrecord.query_edges(query8) == (161, 13)

        def query9(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexOperand:
            edge.index().is_in([160, 161])
            attribute = edge.attributes().max().max()
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_edges(query9) == (161, 13)


class TestEdgeSingleAttributeWithIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 12: "value4"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {" Hello ": "value5"})])

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " Hello "))]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_int()
            return attribute

        assert sorted(
            self.medrecord.query_edges(query), key=operator.itemgetter(0)
        ) == [("value1", (161, 12))]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.greater_than(11)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.greater_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.less_than(13)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.less_than_or_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.not_equal_to(11)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_in([12])
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.is_not_in([11])
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 14))]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.subtract(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 10))]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 24))]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 0))]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 144))]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.subtract(13)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 1))]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " Hello "))]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " Hello "))]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " Hello "))]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, "Hello"))]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, "Hello "))]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " Hello"))]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " He"))]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " hello "))]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .max()
            )
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", (162, " HELLO "))]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.either_or(
                lambda attr: attr.equal_to(12),
                lambda attr: attr.equal_to(11),
            )
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            attribute.exclude(lambda attr: attr.equal_to(11))
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]

    def test_ungroup(self) -> None:
        def query(edge: EdgeOperand) -> EdgeMultipleAttributesWithIndexOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            return attribute.ungroup()

        assert self.medrecord.query_edges(query) == {161: 12}

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .max()
            )
            clone = attribute.clone()
            attribute.add(2)
            return clone

        assert self.medrecord.query_edges(query) == [("value1", (161, 12))]


class TestEdgeSingleAttributeWithoutIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {" Hello ": "value5"})])

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query) == " Hello "

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.greater_than(5)
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.greater_than_or_equal_to(21)
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.less_than(20)
            return attribute

        assert self.medrecord.query_edges(query) is None

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.less_than_or_equal_to(21)
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.equal_to(21)
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.not_equal_to(12)
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.is_in([21, 22])
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.is_not_in([21])
            return attribute

        assert self.medrecord.query_edges(query) is None

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query) == 23

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.subtract(1)
            return attribute

        assert self.medrecord.query_edges(query) == 20

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.multiply(2)
            return attribute

        assert self.medrecord.query_edges(query) == 42

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query) == 1

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.power(2)
            return attribute

        assert self.medrecord.query_edges(query) == 441

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.subtract(23)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query) == 2

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_edges(query) == " Hello "

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_edges(query) == " Hello "

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_edges(query) == " Hello "

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query) == "Hello"

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query) == "Hello "

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query) == " Hello"

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query) == " He"

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query) == " hello "

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(" Hello ")
            attribute = edge.attributes().sum().sum()
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query) == " HELLO "

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.either_or(
                lambda attr: attr.equal_to(21),
                lambda attr: attr.equal_to(12),
            )
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            attribute.exclude(lambda attr: attr.equal_to(12))
            return attribute

        assert self.medrecord.query_edges(query) == 21

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexOperand:
            edge.has_attribute(10)
            attribute = edge.attributes().sum().sum()
            clone = attribute.clone()
            attribute.add(2)
            return clone

        assert self.medrecord.query_edges(query) == 21


class TestEdgeSingleAttributeWithoutIndexGroupOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()
        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 11: "value2"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {10: "value1", 12: "value4"})])
        self.medrecord.add_edges([("pat_1", "pat_2", {" Hello ": "value5"})])

    def test_is_string(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.is_string()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " Hello ")]

    def test_is_int(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.is_int()
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_greater_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.greater_than(1)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_greater_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.greater_than_or_equal_to(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_less_than(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.less_than(3)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_less_than_or_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.less_than_or_equal_to(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.equal_to(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_not_equal_to(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.not_equal_to(1)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_is_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.is_in([2, 3])
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_is_not_in(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.is_not_in([1])
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_add(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.add(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 4)]

    def test_subtract(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.subtract(1)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 1)]

    def test_multiply(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.multiply(3)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 6)]

    def test_modulo(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.modulo(2)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 0)]

    def test_power(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.power(3)
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 8)]

    def test_absolute(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.subtract(5)
            attribute.absolute()
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 3)]

    def test_starts_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.starts_with(" H")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " Hello ")]

    def test_ends_with(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.ends_with("o ")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " Hello ")]

    def test_contains(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.contains("ell")
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " Hello ")]

    def test_trim(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", "Hello")]

    def test_trim_start(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim_start()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", "Hello ")]

    def test_trim_end(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.trim_end()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " Hello")]

    def test_slice(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.slice(0, 3)
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " He")]

    def test_lowercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.lowercase()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " hello ")]

    def test_uppercase(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(" Hello ")
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(" Hello "))
                .attributes()
                .max()
                .sum()
            )
            attribute.uppercase()
            return attribute

        assert self.medrecord.query_edges(query) == [("value5", " HELLO ")]

    def test_either_or(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.either_or(
                lambda attr: attr.equal_to(2),
                lambda attr: attr.equal_to(1),
            )
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_exclude(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            attribute.exclude(lambda attr: attr.equal_to(1))
            return attribute

        assert self.medrecord.query_edges(query) == [("value1", 2)]

    def test_clone(self) -> None:
        def query(edge: EdgeOperand) -> EdgeSingleAttributeWithoutIndexGroupOperand:
            edge.has_attribute(10)
            attribute = (
                edge.group_by(EdgeOperandGroupDiscriminator.Attribute(10))
                .attributes()
                .max()
                .count()
            )
            clone = attribute.clone()
            attribute.add(10)
            return clone

        assert self.medrecord.query_edges(query) == [("value1", 2)]


if __name__ == "__main__":
    pytest.main(["-v", __file__])

# ruff: noqa: D102, D101

"""Query API for MedRecord."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeAlias, Union

from medmodels._medmodels import (
    EdgeOperandGroupDiscriminator,
    NodeOperandGroupDiscriminator,
    PyEdgeAttributesTreeGroupOperand,
    PyEdgeAttributesTreeOperand,
    PyEdgeDirection,
    PyEdgeGroupOperand,
    PyEdgeIndexGroupOperand,
    PyEdgeIndexOperand,
    PyEdgeIndicesGroupOperand,
    PyEdgeIndicesOperand,
    PyEdgeMultipleAttributesWithIndexGroupOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand,
    PyEdgeMultipleValuesWithIndexGroupOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    PyEdgeMultipleValuesWithoutIndexOperand,
    PyEdgeOperand,
    PyEdgeSingleAttributeWithIndexGroupOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithoutIndexGroupOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeSingleValueWithIndexGroupOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeSingleValueWithoutIndexGroupOperand,
    PyEdgeSingleValueWithoutIndexOperand,
    PyNodeAttributesTreeGroupOperand,
    PyNodeAttributesTreeOperand,
    PyNodeGroupOperand,
    PyNodeIndexGroupOperand,
    PyNodeIndexOperand,
    PyNodeIndicesGroupOperand,
    PyNodeIndicesOperand,
    PyNodeMultipleAttributesWithIndexGroupOperand,
    PyNodeMultipleAttributesWithIndexOperand,
    PyNodeMultipleAttributesWithoutIndexOperand,
    PyNodeMultipleValuesWithIndexGroupOperand,
    PyNodeMultipleValuesWithIndexOperand,
    PyNodeMultipleValuesWithoutIndexOperand,
    PyNodeOperand,
    PyNodeSingleAttributeWithIndexGroupOperand,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeSingleAttributeWithoutIndexGroupOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeSingleValueWithIndexGroupOperand,
    PyNodeSingleValueWithIndexOperand,
    PyNodeSingleValueWithoutIndexGroupOperand,
    PyNodeSingleValueWithoutIndexOperand,
)
from medmodels.medrecord.types import (
    EdgeIndex,
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)

GroupKey: TypeAlias = Union[
    NodeIndex, MedRecordValue, Optional[MedRecordValue], Tuple["GroupKey", "GroupKey"]
]


PyQueryReturnOperand: TypeAlias = Union[
    PyNodeAttributesTreeOperand,
    PyNodeAttributesTreeGroupOperand,
    PyEdgeAttributesTreeOperand,
    PyEdgeAttributesTreeGroupOperand,
    PyNodeMultipleAttributesWithIndexOperand,
    PyNodeMultipleAttributesWithIndexGroupOperand,
    PyNodeMultipleAttributesWithoutIndexOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithIndexGroupOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeSingleAttributeWithIndexGroupOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeSingleAttributeWithoutIndexGroupOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithIndexGroupOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeSingleAttributeWithoutIndexGroupOperand,
    PyEdgeIndicesOperand,
    PyEdgeIndicesGroupOperand,
    PyEdgeIndexOperand,
    PyEdgeIndexGroupOperand,
    PyNodeIndicesOperand,
    PyNodeIndicesGroupOperand,
    PyNodeIndexOperand,
    PyNodeIndexGroupOperand,
    PyNodeMultipleValuesWithIndexOperand,
    PyNodeMultipleValuesWithIndexGroupOperand,
    PyNodeMultipleValuesWithoutIndexOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    PyEdgeMultipleValuesWithIndexGroupOperand,
    PyEdgeMultipleValuesWithoutIndexOperand,
    PyNodeSingleValueWithIndexOperand,
    PyNodeSingleValueWithIndexGroupOperand,
    PyNodeSingleValueWithoutIndexOperand,
    PyNodeSingleValueWithoutIndexGroupOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeSingleValueWithIndexGroupOperand,
    PyEdgeSingleValueWithoutIndexOperand,
    PyEdgeSingleValueWithoutIndexGroupOperand,
    Sequence["PyQueryReturnOperand"],
]

#: A type alias for a query return operand.
QueryReturnOperand: TypeAlias = Union[
    "NodeAttributesTreeOperand",
    "NodeAttributesTreeGroupOperand",
    "EdgeAttributesTreeOperand",
    "EdgeAttributesTreeGroupOperand",
    "NodeMultipleAttributesWithIndexOperand",
    "NodeMultipleAttributesWithIndexGroupOperand",
    "NodeMultipleAttributesWithoutIndexOperand",
    "EdgeMultipleAttributesWithIndexOperand",
    "EdgeMultipleAttributesWithIndexGroupOperand",
    "EdgeMultipleAttributesWithoutIndexOperand",
    "NodeSingleAttributeWithIndexOperand",
    "NodeSingleAttributeWithIndexGroupOperand",
    "NodeSingleAttributeWithoutIndexOperand",
    "NodeSingleAttributeWithoutIndexGroupOperand",
    "EdgeSingleAttributeWithIndexOperand",
    "EdgeSingleAttributeWithIndexGroupOperand",
    "EdgeSingleAttributeWithoutIndexOperand",
    "EdgeSingleAttributeWithoutIndexGroupOperand",
    "EdgeIndicesOperand",
    "EdgeIndicesGroupOperand",
    "EdgeIndexOperand",
    "EdgeIndexGroupOperand",
    "NodeIndicesOperand",
    "NodeIndicesGroupOperand",
    "NodeIndexOperand",
    "NodeIndexGroupOperand",
    "NodeMultipleValuesWithIndexOperand",
    "NodeMultipleValuesWithIndexGroupOperand",
    "NodeMultipleValuesWithoutIndexOperand",
    "EdgeMultipleValuesWithIndexOperand",
    "EdgeMultipleValuesWithIndexGroupOperand",
    "EdgeMultipleValuesWithoutIndexOperand",
    "NodeSingleValueWithIndexOperand",
    "NodeSingleValueWithIndexGroupOperand",
    "NodeSingleValueWithoutIndexOperand",
    "NodeSingleValueWithoutIndexGroupOperand",
    "EdgeSingleValueWithIndexOperand",
    "EdgeSingleValueWithIndexGroupOperand",
    "EdgeSingleValueWithoutIndexOperand",
    "EdgeSingleValueWithoutIndexGroupOperand",
    Sequence["QueryReturnOperand"],
]

NodeAttributesTreeQueryResult: TypeAlias = Dict[NodeIndex, List[MedRecordAttribute]]
NodeAttributesTreeGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeAttributesTreeQueryResult]
]
EdgeAttributesTreeQueryResult: TypeAlias = Dict[EdgeIndex, List[MedRecordAttribute]]
EdgeAttributesTreeGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeAttributesTreeQueryResult]
]

NodeMultipleAttributesWithIndexQueryResult: TypeAlias = Dict[
    NodeIndex, MedRecordAttribute
]
NodeMultipleAttributesWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeMultipleAttributesWithIndexQueryResult]
]
NodeMultipleAttributesWithoutIndexQueryResult: TypeAlias = List[MedRecordAttribute]
EdgeMultipleAttributesWithIndexQueryResult: TypeAlias = Dict[
    EdgeIndex, MedRecordAttribute
]
EdgeMultipleAttributesWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeMultipleAttributesWithIndexQueryResult]
]
EdgeMultipleAttributesWithoutIndexQueryResult: TypeAlias = List[MedRecordAttribute]

NodeSingleAttributeWithIndexQueryResult: TypeAlias = Tuple[
    NodeIndex, MedRecordAttribute
]
NodeSingleAttributeWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeSingleAttributeWithIndexQueryResult]
]
NodeSingleAttributeWithoutIndexQueryResult: TypeAlias = MedRecordAttribute
NodeSingleAttributeWithoutIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeSingleAttributeWithoutIndexQueryResult]
]
EdgeSingleAttributeWithIndexQueryResult: TypeAlias = Tuple[
    EdgeIndex, MedRecordAttribute
]
EdgeSingleAttributeWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeSingleAttributeWithIndexQueryResult]
]
EdgeSingleAttributeWithoutIndexQueryResult: TypeAlias = MedRecordAttribute
EdgeSingleAttributeWithoutIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeSingleAttributeWithoutIndexQueryResult]
]

EdgeIndicesQueryResult: TypeAlias = List[EdgeIndex]
EdgeIndicesGroupQueryResult: TypeAlias = List[Tuple[GroupKey, EdgeIndicesQueryResult]]

EdgeIndexQueryResult: TypeAlias = Optional[EdgeIndex]
EdgeIndexGroupQueryResult: TypeAlias = List[Tuple[GroupKey, EdgeIndexQueryResult]]

NodeIndicesQueryResult: TypeAlias = List[NodeIndex]
NodeIndicesGroupQueryResult: TypeAlias = List[Tuple[GroupKey, NodeIndicesQueryResult]]

NodeIndexQueryResult: TypeAlias = Optional[NodeIndex]
NodeIndexGroupQueryResult: TypeAlias = List[Tuple[GroupKey, NodeIndexQueryResult]]

NodeMultipleValuesWithIndexQueryResult: TypeAlias = Dict[NodeIndex, MedRecordValue]
NodeMultipleValuesWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeMultipleValuesWithIndexQueryResult]
]
NodeMultipleValuesWithoutIndexQueryResult: TypeAlias = List[MedRecordValue]
EdgeMultipleValuesWithIndexQueryResult: TypeAlias = Dict[EdgeIndex, MedRecordValue]
EdgeMultipleValuesWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeMultipleValuesWithIndexQueryResult]
]
EdgeMultipleValuesWithoutIndexQueryResult: TypeAlias = List[MedRecordValue]

NodeSingleValueWithIndexQueryResult: TypeAlias = Tuple[NodeIndex, MedRecordValue]
NodeSingleValueWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeSingleValueWithIndexQueryResult]
]
NodeSingleValueWithoutIndexQueryResult: TypeAlias = MedRecordValue
NodeSingleValueWithoutIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, NodeSingleValueWithoutIndexQueryResult]
]
EdgeSingleValueWithIndexQueryResult: TypeAlias = Tuple[EdgeIndex, MedRecordValue]
EdgeSingleValueWithIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeSingleValueWithIndexQueryResult]
]
EdgeSingleValueWithoutIndexQueryResult: TypeAlias = MedRecordValue
EdgeSingleValueWithoutIndexGroupQueryResult: TypeAlias = List[
    Tuple[GroupKey, EdgeSingleValueWithoutIndexQueryResult]
]

#: A type alias for a query result.
QueryResult: TypeAlias = Union[
    NodeAttributesTreeQueryResult,
    NodeAttributesTreeGroupQueryResult,
    EdgeAttributesTreeQueryResult,
    EdgeAttributesTreeGroupQueryResult,
    NodeMultipleAttributesWithIndexQueryResult,
    NodeMultipleAttributesWithIndexGroupQueryResult,
    NodeMultipleAttributesWithoutIndexQueryResult,
    EdgeMultipleAttributesWithIndexQueryResult,
    EdgeMultipleAttributesWithIndexGroupQueryResult,
    EdgeMultipleAttributesWithoutIndexQueryResult,
    NodeSingleAttributeWithIndexQueryResult,
    NodeSingleAttributeWithIndexGroupQueryResult,
    NodeSingleAttributeWithoutIndexQueryResult,
    NodeSingleAttributeWithoutIndexGroupQueryResult,
    EdgeSingleAttributeWithIndexQueryResult,
    EdgeSingleAttributeWithIndexGroupQueryResult,
    EdgeSingleAttributeWithoutIndexQueryResult,
    EdgeIndicesQueryResult,
    EdgeIndicesGroupQueryResult,
    EdgeIndexQueryResult,
    EdgeIndexGroupQueryResult,
    NodeIndicesQueryResult,
    NodeIndicesGroupQueryResult,
    NodeIndexQueryResult,
    NodeIndexGroupQueryResult,
    NodeMultipleValuesWithIndexQueryResult,
    NodeMultipleValuesWithIndexGroupQueryResult,
    NodeMultipleValuesWithoutIndexQueryResult,
    EdgeMultipleValuesWithIndexQueryResult,
    EdgeMultipleValuesWithIndexGroupQueryResult,
    EdgeMultipleValuesWithoutIndexQueryResult,
    NodeSingleValueWithIndexQueryResult,
    NodeSingleValueWithIndexGroupQueryResult,
    NodeSingleValueWithoutIndexQueryResult,
    NodeSingleValueWithoutIndexGroupQueryResult,
    EdgeSingleValueWithIndexQueryResult,
    EdgeSingleValueWithIndexGroupQueryResult,
    EdgeSingleValueWithoutIndexQueryResult,
    EdgeSingleValueWithoutIndexGroupQueryResult,
    List["QueryResult"],
]

NodeQuery: TypeAlias = Callable[["NodeOperand"], QueryReturnOperand]
NodeQueryComponent: TypeAlias = Callable[["NodeOperand"], None]
NodeIndicesQuery: TypeAlias = Callable[["NodeOperand"], "NodeIndicesOperand"]
NodeIndexQuery: TypeAlias = Callable[["NodeOperand"], "NodeIndexOperand"]

EdgeQuery: TypeAlias = Callable[["EdgeOperand"], QueryReturnOperand]
EdgeQueryComponent: TypeAlias = Callable[["EdgeOperand"], None]
EdgeIndicesQuery: TypeAlias = Callable[["EdgeOperand"], "EdgeIndicesOperand"]
EdgeIndexQuery: TypeAlias = Callable[["EdgeOperand"], "EdgeIndexOperand"]

SingleValueComparisonOperand: TypeAlias = Union[
    MedRecordValue,
    "NodeSingleValueWithIndexOperand",
    "NodeSingleValueWithoutIndexOperand",
    "EdgeSingleValueWithIndexOperand",
    "EdgeSingleValueWithoutIndexOperand",
]
SingleValueArithmeticOperand: TypeAlias = SingleValueComparisonOperand
MultipleValuesComparisonOperand: TypeAlias = Union[
    List[MedRecordValue],
    "NodeMultipleValuesWithIndexOperand",
    "NodeMultipleValuesWithoutIndexOperand",
    "EdgeMultipleValuesWithIndexOperand",
    "EdgeMultipleValuesWithoutIndexOperand",
]


def _py_single_value_comparison_operand_from_single_value_comparison_operand(
    single_value_comparison_operand: SingleValueComparisonOperand,
) -> Union[
    MedRecordValue,
    PyNodeSingleValueWithIndexOperand,
    PyNodeSingleValueWithoutIndexOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeSingleValueWithoutIndexOperand,
]:
    if isinstance(single_value_comparison_operand, NodeSingleValueWithIndexOperand):
        return single_value_comparison_operand._single_value_operand
    if isinstance(single_value_comparison_operand, NodeSingleValueWithoutIndexOperand):
        return single_value_comparison_operand._single_value_operand
    if isinstance(single_value_comparison_operand, EdgeSingleValueWithIndexOperand):
        return single_value_comparison_operand._single_value_operand
    if isinstance(single_value_comparison_operand, EdgeSingleValueWithoutIndexOperand):
        return single_value_comparison_operand._single_value_operand
    return single_value_comparison_operand


def _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
    multiple_values_comparison_operand: MultipleValuesComparisonOperand,
) -> Union[
    List[MedRecordValue],
    PyNodeMultipleValuesWithIndexOperand,
    PyNodeMultipleValuesWithoutIndexOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    PyEdgeMultipleValuesWithoutIndexOperand,
]:
    if isinstance(
        multiple_values_comparison_operand, NodeMultipleValuesWithIndexOperand
    ):
        return multiple_values_comparison_operand._multiple_values_operand
    if isinstance(
        multiple_values_comparison_operand, NodeMultipleValuesWithoutIndexOperand
    ):
        return multiple_values_comparison_operand._multiple_values_operand
    if isinstance(
        multiple_values_comparison_operand, EdgeMultipleValuesWithIndexOperand
    ):
        return multiple_values_comparison_operand._multiple_values_operand
    if isinstance(
        multiple_values_comparison_operand, EdgeMultipleValuesWithoutIndexOperand
    ):
        return multiple_values_comparison_operand._multiple_values_operand
    return multiple_values_comparison_operand


SingleAttributeComparisonOperand: TypeAlias = Union[
    MedRecordAttribute,
    "NodeSingleAttributeWithIndexOperand",
    "NodeSingleAttributeWithoutIndexOperand",
    "EdgeSingleAttributeWithIndexOperand",
    "EdgeSingleAttributeWithoutIndexOperand",
]
SingleAttributeArithmeticOperand: TypeAlias = SingleAttributeComparisonOperand
MultipleAttributesComparisonOperand: TypeAlias = Union[
    List[MedRecordAttribute],
    "NodeMultipleAttributesWithIndexOperand",
    "NodeMultipleAttributesWithoutIndexOperand",
    "EdgeMultipleAttributesWithIndexOperand",
    "EdgeMultipleAttributesWithoutIndexOperand",
]


def _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
    single_attribute_comparison_operand: SingleAttributeComparisonOperand,
) -> Union[
    MedRecordAttribute,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
]:
    if isinstance(
        single_attribute_comparison_operand, NodeSingleAttributeWithIndexOperand
    ):
        return single_attribute_comparison_operand._single_attribute_operand
    if isinstance(
        single_attribute_comparison_operand, NodeSingleAttributeWithoutIndexOperand
    ):
        return single_attribute_comparison_operand._single_attribute_operand
    if isinstance(
        single_attribute_comparison_operand, EdgeSingleAttributeWithIndexOperand
    ):
        return single_attribute_comparison_operand._single_attribute_operand
    if isinstance(
        single_attribute_comparison_operand, EdgeSingleAttributeWithoutIndexOperand
    ):
        return single_attribute_comparison_operand._single_attribute_operand
    return single_attribute_comparison_operand


def _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
    multiple_attributes_comparison_operand: MultipleAttributesComparisonOperand,
) -> Union[
    List[MedRecordAttribute],
    PyNodeMultipleAttributesWithIndexOperand,
    PyNodeMultipleAttributesWithoutIndexOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand,
]:
    if isinstance(
        multiple_attributes_comparison_operand, NodeMultipleAttributesWithIndexOperand
    ):
        return multiple_attributes_comparison_operand._multiple_attributes_operand
    if isinstance(
        multiple_attributes_comparison_operand,
        NodeMultipleAttributesWithoutIndexOperand,
    ):
        return multiple_attributes_comparison_operand._multiple_attributes_operand
    if isinstance(
        multiple_attributes_comparison_operand, EdgeMultipleAttributesWithIndexOperand
    ):
        return multiple_attributes_comparison_operand._multiple_attributes_operand
    if isinstance(
        multiple_attributes_comparison_operand,
        EdgeMultipleAttributesWithoutIndexOperand,
    ):
        return multiple_attributes_comparison_operand._multiple_attributes_operand
    return multiple_attributes_comparison_operand


NodeIndexComparisonOperand: TypeAlias = Union["NodeIndexOperand", NodeIndex]
NodeIndexArithmeticOperand: TypeAlias = NodeIndexComparisonOperand
NodeIndicesComparisonOperand: TypeAlias = Union["NodeIndicesOperand", List[NodeIndex]]


def _py_node_index_comparison_operand_from_node_index_comparison_operand(
    node_index_comparison_operand: NodeIndexComparisonOperand,
) -> Union[NodeIndex, PyNodeIndexOperand]:
    if isinstance(node_index_comparison_operand, NodeIndexOperand):
        return node_index_comparison_operand._node_index_operand
    return node_index_comparison_operand


def _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
    node_indices_comparison_operand: NodeIndicesComparisonOperand,
) -> Union[List[NodeIndex], PyNodeIndicesOperand]:
    if isinstance(node_indices_comparison_operand, NodeIndicesOperand):
        return node_indices_comparison_operand._node_indices_operand
    return node_indices_comparison_operand


EdgeIndexComparisonOperand: TypeAlias = Union[
    "EdgeIndexOperand",
    EdgeIndex,
]
EdgeIndexArithmeticOperand: TypeAlias = EdgeIndexComparisonOperand
EdgeIndicesComparisonOperand: TypeAlias = Union[
    "EdgeIndicesOperand",
    List[EdgeIndex],
]


def _py_edge_index_comparison_operand_from_edge_index_comparison_operand(
    edge_index_comparison_operand: EdgeIndexComparisonOperand,
) -> Union[EdgeIndex, PyEdgeIndexOperand]:
    if isinstance(edge_index_comparison_operand, EdgeIndexOperand):
        return edge_index_comparison_operand._edge_index_operand
    return edge_index_comparison_operand


def _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
    edge_indices_comparison_operand: EdgeIndicesComparisonOperand,
) -> Union[List[EdgeIndex], PyEdgeIndicesOperand]:
    if isinstance(edge_indices_comparison_operand, EdgeIndicesOperand):
        return edge_indices_comparison_operand._edge_indices_operand
    return edge_indices_comparison_operand


class EdgeDirection(Enum):
    """Enumeration of edge directions."""

    INCOMING = 0
    OUTGOING = 1
    BOTH = 2

    def _into_py_edge_direction(self) -> PyEdgeDirection:
        return (
            PyEdgeDirection.Incoming
            if self == EdgeDirection.INCOMING
            else PyEdgeDirection.Outgoing
            if self == EdgeDirection.OUTGOING
            else PyEdgeDirection.Both
        )


class NodeOperand:
    _node_operand: PyNodeOperand

    def attribute(
        self, attribute: MedRecordAttribute
    ) -> NodeMultipleValuesWithIndexOperand:
        return NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._node_operand.attribute(attribute)
        )

    def attributes(self) -> NodeAttributesTreeOperand:
        return NodeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._node_operand.attributes()
        )

    def index(self) -> NodeIndicesOperand:
        return NodeIndicesOperand._from_py_node_indices_operand(
            self._node_operand.index()
        )

    def in_group(self, group: Union[Group, List[Group]]) -> None:
        self._node_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None:
        self._node_operand.has_attribute(attribute)

    def edges(self, direction: EdgeDirection = EdgeDirection.BOTH) -> EdgeOperand:
        return EdgeOperand._from_py_edge_operand(
            self._node_operand.edges(direction._into_py_edge_direction())
        )

    def neighbors(
        self, edge_direction: EdgeDirection = EdgeDirection.BOTH
    ) -> NodeOperand:
        return NodeOperand._from_py_node_operand(
            self._node_operand.neighbors(edge_direction._into_py_edge_direction())
        )

    def either_or(
        self,
        either: NodeQueryComponent,
        or_: NodeQueryComponent,
    ) -> None:
        self._node_operand.either_or(
            lambda node: either(NodeOperand._from_py_node_operand(node)),
            lambda node: or_(NodeOperand._from_py_node_operand(node)),
        )

    def exclude(self, query: NodeQueryComponent) -> None:
        self._node_operand.exclude(
            lambda node: query(NodeOperand._from_py_node_operand(node))
        )

    def group_by(
        self, discriminator: NodeOperandGroupDiscriminator
    ) -> NodeGroupOperand:
        return NodeGroupOperand._from_py_node_group_operand(
            self._node_operand.group_by(discriminator)
        )

    def clone(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._node_operand.deep_clone())

    @classmethod
    def _from_py_node_operand(cls, py_node_operand: PyNodeOperand) -> NodeOperand:
        node_operand = cls()
        node_operand._node_operand = py_node_operand
        return node_operand


class NodeGroupOperand:
    _node_operand: PyNodeGroupOperand

    def attribute(
        self, attribute: MedRecordAttribute
    ) -> NodeMultipleValuesWithIndexGroupOperand:
        return NodeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._node_operand.attribute(attribute)
        )

    def attributes(self) -> NodeAttributesTreeGroupOperand:
        return NodeAttributesTreeGroupOperand._from_py_attributes_tree_operand(
            self._node_operand.attributes()
        )

    def index(self) -> NodeIndicesGroupOperand:
        return NodeIndicesGroupOperand._from_py_node_indices_operand(
            self._node_operand.index()
        )

    def in_group(self, group: Union[Group, List[Group]]) -> None:
        self._node_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None:
        self._node_operand.has_attribute(attribute)

    def edges(self, direction: EdgeDirection = EdgeDirection.BOTH) -> EdgeGroupOperand:
        return EdgeGroupOperand._from_py_edge_group_operand(
            self._node_operand.edges(direction._into_py_edge_direction())
        )

    def neighbors(
        self, edge_direction: EdgeDirection = EdgeDirection.BOTH
    ) -> NodeGroupOperand:
        return NodeGroupOperand._from_py_node_group_operand(
            self._node_operand.neighbors(edge_direction._into_py_edge_direction())
        )

    def either_or(
        self,
        either: NodeQueryComponent,
        or_: NodeQueryComponent,
    ) -> None:
        self._node_operand.either_or(
            lambda node: either(NodeOperand._from_py_node_operand(node)),
            lambda node: or_(NodeOperand._from_py_node_operand(node)),
        )

    def exclude(self, query: NodeQueryComponent) -> None:
        self._node_operand.exclude(
            lambda node: query(NodeOperand._from_py_node_operand(node))
        )

    def clone(self) -> NodeGroupOperand:
        return NodeGroupOperand._from_py_node_group_operand(
            self._node_operand.deep_clone()
        )

    @classmethod
    def _from_py_node_group_operand(
        cls, py_node_operand: PyNodeGroupOperand
    ) -> NodeGroupOperand:
        node_operand = cls()
        node_operand._node_operand = py_node_operand
        return node_operand


class EdgeOperand:
    _edge_operand: PyEdgeOperand

    def attribute(
        self, attribute: MedRecordAttribute
    ) -> EdgeMultipleValuesWithIndexOperand:
        return EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._edge_operand.attribute(attribute)
        )

    def attributes(self) -> EdgeAttributesTreeOperand:
        return EdgeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._edge_operand.attributes()
        )

    def index(self) -> EdgeIndicesOperand:
        return EdgeIndicesOperand._from_edge_indices_operand(self._edge_operand.index())

    def in_group(self, group: Union[Group, List[Group]]) -> None:
        self._edge_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None:
        self._edge_operand.has_attribute(attribute)

    def source_node(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._edge_operand.source_node())

    def target_node(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._edge_operand.target_node())

    def either_or(
        self,
        either: EdgeQueryComponent,
        or_: EdgeQueryComponent,
    ) -> None:
        self._edge_operand.either_or(
            lambda edge: either(EdgeOperand._from_py_edge_operand(edge)),
            lambda edge: or_(EdgeOperand._from_py_edge_operand(edge)),
        )

    def exclude(self, query: EdgeQueryComponent) -> None:
        self._edge_operand.exclude(
            lambda edge: query(EdgeOperand._from_py_edge_operand(edge))
        )

    def group_by(
        self, discriminator: EdgeOperandGroupDiscriminator
    ) -> EdgeGroupOperand:
        return EdgeGroupOperand._from_py_edge_group_operand(
            self._edge_operand.group_by(discriminator)
        )

    def clone(self) -> EdgeOperand:
        return EdgeOperand._from_py_edge_operand(self._edge_operand.deep_clone())

    @classmethod
    def _from_py_edge_operand(cls, py_edge_operand: PyEdgeOperand) -> EdgeOperand:
        edge_operand = cls()
        edge_operand._edge_operand = py_edge_operand
        return edge_operand


class EdgeGroupOperand:
    _edge_operand: PyEdgeGroupOperand

    def attribute(
        self, attribute: MedRecordAttribute
    ) -> EdgeMultipleValuesWithIndexGroupOperand:
        return EdgeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._edge_operand.attribute(attribute)
        )

    def attributes(self) -> EdgeAttributesTreeGroupOperand:
        return EdgeAttributesTreeGroupOperand._from_py_attributes_tree_operand(
            self._edge_operand.attributes()
        )

    def index(self) -> EdgeIndicesGroupOperand:
        return EdgeIndicesGroupOperand._from_edge_indices_operand(
            self._edge_operand.index()
        )

    def in_group(self, group: Union[Group, List[Group]]) -> None:
        self._edge_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None:
        self._edge_operand.has_attribute(attribute)

    def source_node(self) -> NodeGroupOperand:
        return NodeGroupOperand._from_py_node_group_operand(
            self._edge_operand.source_node()
        )

    def target_node(self) -> NodeGroupOperand:
        return NodeGroupOperand._from_py_node_group_operand(
            self._edge_operand.target_node()
        )

    def either_or(
        self,
        either: EdgeQueryComponent,
        or_: EdgeQueryComponent,
    ) -> None:
        self._edge_operand.either_or(
            lambda edge: either(EdgeOperand._from_py_edge_operand(edge)),
            lambda edge: or_(EdgeOperand._from_py_edge_operand(edge)),
        )

    def exclude(self, query: EdgeQueryComponent) -> None:
        self._edge_operand.exclude(
            lambda edge: query(EdgeOperand._from_py_edge_operand(edge))
        )

    def clone(self) -> EdgeGroupOperand:
        return EdgeGroupOperand._from_py_edge_group_operand(
            self._edge_operand.deep_clone()
        )

    @classmethod
    def _from_py_edge_group_operand(
        cls, py_edge_operand: PyEdgeGroupOperand
    ) -> EdgeGroupOperand:
        edge_operand = cls()
        edge_operand._edge_operand = py_edge_operand
        return edge_operand


class NodeMultipleValuesWithIndexOperand:
    _multiple_values_operand: PyNodeMultipleValuesWithIndexOperand

    def max(self) -> NodeSingleValueWithIndexOperand:
        return NodeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> NodeSingleValueWithIndexOperand:
        return NodeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> NodeSingleValueWithIndexOperand:
        return NodeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleValuesWithIndexOperand], None],
        or_: Callable[[NodeMultipleValuesWithIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleValuesWithIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def clone(self) -> NodeMultipleValuesWithIndexOperand:
        return NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyNodeMultipleValuesWithIndexOperand
    ) -> NodeMultipleValuesWithIndexOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class NodeMultipleValuesWithIndexGroupOperand:
    _multiple_values_operand: PyNodeMultipleValuesWithIndexGroupOperand

    def max(self) -> NodeSingleValueWithIndexGroupOperand:
        return NodeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> NodeSingleValueWithIndexGroupOperand:
        return NodeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> NodeSingleValueWithIndexGroupOperand:
        return NodeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleValuesWithIndexOperand], None],
        or_: Callable[[NodeMultipleValuesWithIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleValuesWithIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def ungroup(self) -> NodeMultipleValuesWithIndexOperand:
        return NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.ungroup()
        )

    def clone(self) -> NodeMultipleValuesWithIndexGroupOperand:
        return NodeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyNodeMultipleValuesWithIndexGroupOperand
    ) -> NodeMultipleValuesWithIndexGroupOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class NodeMultipleValuesWithoutIndexOperand:
    _multiple_values_operand: PyNodeMultipleValuesWithoutIndexOperand

    def max(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleValuesWithoutIndexOperand], None],
        or_: Callable[[NodeMultipleValuesWithoutIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                NodeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                NodeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleValuesWithoutIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                NodeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def clone(self) -> NodeMultipleValuesWithoutIndexOperand:
        return NodeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyNodeMultipleValuesWithoutIndexOperand
    ) -> NodeMultipleValuesWithoutIndexOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class EdgeMultipleValuesWithIndexOperand:
    _multiple_values_operand: PyEdgeMultipleValuesWithIndexOperand

    def max(self) -> EdgeSingleValueWithIndexOperand:
        return EdgeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> EdgeSingleValueWithIndexOperand:
        return EdgeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> EdgeSingleValueWithIndexOperand:
        return EdgeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleValuesWithIndexOperand], None],
        or_: Callable[[EdgeMultipleValuesWithIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleValuesWithIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def clone(self) -> EdgeMultipleValuesWithIndexOperand:
        return EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyEdgeMultipleValuesWithIndexOperand
    ) -> EdgeMultipleValuesWithIndexOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class EdgeMultipleValuesWithIndexGroupOperand:
    _multiple_values_operand: PyEdgeMultipleValuesWithIndexGroupOperand

    def max(self) -> EdgeSingleValueWithIndexGroupOperand:
        return EdgeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> EdgeSingleValueWithIndexGroupOperand:
        return EdgeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> EdgeSingleValueWithIndexGroupOperand:
        return EdgeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleValuesWithIndexOperand], None],
        or_: Callable[[EdgeMultipleValuesWithIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleValuesWithIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def ungroup(self) -> EdgeMultipleValuesWithIndexOperand:
        return EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.ungroup()
        )

    def clone(self) -> EdgeMultipleValuesWithIndexGroupOperand:
        return EdgeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyEdgeMultipleValuesWithIndexGroupOperand
    ) -> EdgeMultipleValuesWithIndexGroupOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class EdgeMultipleValuesWithoutIndexOperand:
    _multiple_values_operand: PyEdgeMultipleValuesWithoutIndexOperand

    def max(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def random(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._multiple_values_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_values_operand.is_string()

    def is_int(self) -> None:
        self._multiple_values_operand.is_int()

    def is_float(self) -> None:
        self._multiple_values_operand.is_float()

    def is_bool(self) -> None:
        self._multiple_values_operand.is_bool()

    def is_datetime(self) -> None:
        self._multiple_values_operand.is_datetime()

    def is_duration(self) -> None:
        self._multiple_values_operand.is_duration()

    def is_null(self) -> None:
        self._multiple_values_operand.is_null()

    def is_max(self) -> None:
        self._multiple_values_operand.is_max()

    def is_min(self) -> None:
        self._multiple_values_operand.is_min()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._multiple_values_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._multiple_values_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._multiple_values_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._multiple_values_operand.round()

    def ceil(self) -> None:
        self._multiple_values_operand.ceil()

    def floor(self) -> None:
        self._multiple_values_operand.floor()

    def absolute(self) -> None:
        self._multiple_values_operand.abs()

    def sqrt(self) -> None:
        self._multiple_values_operand.sqrt()

    def trim(self) -> None:
        self._multiple_values_operand.trim()

    def trim_start(self) -> None:
        self._multiple_values_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_values_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_values_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_values_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_values_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleValuesWithoutIndexOperand], None],
        or_: Callable[[EdgeMultipleValuesWithoutIndexOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                EdgeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
            lambda values: or_(
                EdgeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleValuesWithoutIndexOperand], None]
    ) -> None:
        self._multiple_values_operand.exclude(
            lambda values: query(
                EdgeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
                    values
                )
            )
        )

    def clone(self) -> EdgeMultipleValuesWithoutIndexOperand:
        return EdgeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyEdgeMultipleValuesWithoutIndexOperand
    ) -> EdgeMultipleValuesWithoutIndexOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class NodeSingleValueWithIndexOperand:
    _single_value_operand: PyNodeSingleValueWithIndexOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleValueWithIndexOperand], None],
        or_: Callable[[NodeSingleValueWithIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(self, query: Callable[[NodeSingleValueWithIndexOperand], None]) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            )
        )

    def clone(self) -> NodeSingleValueWithIndexOperand:
        return NodeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyNodeSingleValueWithIndexOperand
    ) -> NodeSingleValueWithIndexOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class NodeSingleValueWithIndexGroupOperand:
    _single_value_operand: PyNodeSingleValueWithIndexGroupOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleValueWithIndexOperand], None],
        or_: Callable[[NodeSingleValueWithIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(self, query: Callable[[NodeSingleValueWithIndexOperand], None]) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                NodeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            )
        )

    def ungroup(self) -> NodeMultipleValuesWithIndexOperand:
        return NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._single_value_operand.ungroup()
        )

    def clone(self) -> NodeSingleValueWithIndexGroupOperand:
        return NodeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyNodeSingleValueWithIndexGroupOperand
    ) -> NodeSingleValueWithIndexGroupOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class NodeSingleValueWithoutIndexOperand:
    _single_value_operand: PyNodeSingleValueWithoutIndexOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleValueWithoutIndexOperand], None],
        or_: Callable[[NodeSingleValueWithoutIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleValueWithoutIndexOperand], None]
    ) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            )
        )

    def clone(self) -> NodeSingleValueWithoutIndexOperand:
        return NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyNodeSingleValueWithoutIndexOperand
    ) -> NodeSingleValueWithoutIndexOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class NodeSingleValueWithoutIndexGroupOperand:
    _single_value_operand: PyNodeSingleValueWithoutIndexGroupOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleValueWithoutIndexOperand], None],
        or_: Callable[[NodeSingleValueWithoutIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleValueWithoutIndexOperand], None]
    ) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                NodeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            )
        )

    def ungroup(self) -> NodeMultipleValuesWithoutIndexOperand:
        return NodeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
            self._single_value_operand.ungroup()
        )

    def clone(self) -> NodeSingleValueWithoutIndexGroupOperand:
        return NodeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyNodeSingleValueWithoutIndexGroupOperand
    ) -> NodeSingleValueWithoutIndexGroupOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class EdgeSingleValueWithIndexOperand:
    _single_value_operand: PyEdgeSingleValueWithIndexOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleValueWithIndexOperand], None],
        or_: Callable[[EdgeSingleValueWithIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(self, query: Callable[[EdgeSingleValueWithIndexOperand], None]) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            )
        )

    def clone(self) -> EdgeSingleValueWithIndexOperand:
        return EdgeSingleValueWithIndexOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyEdgeSingleValueWithIndexOperand
    ) -> EdgeSingleValueWithIndexOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class EdgeSingleValueWithIndexGroupOperand:
    _single_value_operand: PyEdgeSingleValueWithIndexGroupOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleValueWithIndexOperand], None],
        or_: Callable[[EdgeSingleValueWithIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(self, query: Callable[[EdgeSingleValueWithIndexOperand], None]) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                EdgeSingleValueWithIndexOperand._from_py_single_value_operand(value)
            )
        )

    def ungroup(self) -> EdgeMultipleValuesWithIndexOperand:
        return EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._single_value_operand.ungroup()
        )

    def clone(self) -> EdgeSingleValueWithIndexGroupOperand:
        return EdgeSingleValueWithIndexGroupOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyEdgeSingleValueWithIndexGroupOperand
    ) -> EdgeSingleValueWithIndexGroupOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class EdgeSingleValueWithoutIndexOperand:
    _single_value_operand: PyEdgeSingleValueWithoutIndexOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleValueWithoutIndexOperand], None],
        or_: Callable[[EdgeSingleValueWithoutIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleValueWithoutIndexOperand], None]
    ) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            )
        )

    def clone(self) -> EdgeSingleValueWithoutIndexOperand:
        return EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyEdgeSingleValueWithoutIndexOperand
    ) -> EdgeSingleValueWithoutIndexOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class EdgeSingleValueWithoutIndexGroupOperand:
    _single_value_operand: PyEdgeSingleValueWithoutIndexGroupOperand

    def is_string(self) -> None:
        self._single_value_operand.is_string()

    def is_int(self) -> None:
        self._single_value_operand.is_int()

    def is_float(self) -> None:
        self._single_value_operand.is_float()

    def is_bool(self) -> None:
        self._single_value_operand.is_bool()

    def is_datetime(self) -> None:
        self._single_value_operand.is_datetime()

    def is_duration(self) -> None:
        self._single_value_operand.is_duration()

    def is_null(self) -> None:
        self._single_value_operand.is_null()

    def greater_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.greater_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.less_than_or_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def not_equal_to(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.not_equal_to(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def is_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None:
        self._single_value_operand.is_not_in(
            _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
                values
            )
        )

    def starts_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.starts_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def ends_with(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.ends_with(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def contains(self, value: SingleValueComparisonOperand) -> None:
        self._single_value_operand.contains(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def add(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.add(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def subtract(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.sub(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def multiply(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mul(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def divide(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.div(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def modulo(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.mod(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def power(self, value: SingleValueArithmeticOperand) -> None:
        self._single_value_operand.pow(
            _py_single_value_comparison_operand_from_single_value_comparison_operand(
                value
            )
        )

    def round(self) -> None:
        self._single_value_operand.round()

    def ceil(self) -> None:
        self._single_value_operand.ceil()

    def floor(self) -> None:
        self._single_value_operand.floor()

    def absolute(self) -> None:
        self._single_value_operand.abs()

    def sqrt(self) -> None:
        self._single_value_operand.sqrt()

    def trim(self) -> None:
        self._single_value_operand.trim()

    def trim_start(self) -> None:
        self._single_value_operand.trim_start()

    def trim_end(self) -> None:
        self._single_value_operand.trim_end()

    def lowercase(self) -> None:
        self._single_value_operand.lowercase()

    def uppercase(self) -> None:
        self._single_value_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_value_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleValueWithoutIndexOperand], None],
        or_: Callable[[EdgeSingleValueWithoutIndexOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleValueWithoutIndexOperand], None]
    ) -> None:
        self._single_value_operand.exclude(
            lambda value: query(
                EdgeSingleValueWithoutIndexOperand._from_py_single_value_operand(value)
            )
        )

    def ungroup(self) -> EdgeMultipleValuesWithoutIndexOperand:
        return EdgeMultipleValuesWithoutIndexOperand._from_py_multiple_values_operand(
            self._single_value_operand.ungroup()
        )

    def clone(self) -> EdgeSingleValueWithoutIndexGroupOperand:
        return EdgeSingleValueWithoutIndexGroupOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PyEdgeSingleValueWithoutIndexGroupOperand
    ) -> EdgeSingleValueWithoutIndexGroupOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class NodeAttributesTreeOperand:
    _attributes_tree_operand: PyNodeAttributesTreeOperand

    def max(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.max()
            )
        )

    def min(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.min()
            )
        )

    def count(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.count()
            )
        )

    def sum(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.sum()
            )
        )

    def random(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.random()
            )
        )

    def is_string(self) -> None:
        self._attributes_tree_operand.is_string()

    def is_int(self) -> None:
        self._attributes_tree_operand.is_int()

    def is_max(self) -> None:
        self._attributes_tree_operand.is_max()

    def is_min(self) -> None:
        self._attributes_tree_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._attributes_tree_operand.abs()

    def trim(self) -> None:
        self._attributes_tree_operand.trim()

    def trim_start(self) -> None:
        self._attributes_tree_operand.trim_start()

    def trim_end(self) -> None:
        self._attributes_tree_operand.trim_end()

    def lowercase(self) -> None:
        self._attributes_tree_operand.lowercase()

    def uppercase(self) -> None:
        self._attributes_tree_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._attributes_tree_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeAttributesTreeOperand], None],
        or_: Callable[[NodeAttributesTreeOperand], None],
    ) -> None:
        self._attributes_tree_operand.either_or(
            lambda attributes: either(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
            lambda attributes: or_(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
        )

    def exclude(self, query: Callable[[NodeAttributesTreeOperand], None]) -> None:
        self._attributes_tree_operand.exclude(
            lambda attributes: query(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            )
        )

    def clone(self) -> NodeAttributesTreeOperand:
        return NodeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.deep_clone()
        )

    @classmethod
    def _from_py_attributes_tree_operand(
        cls, py_attributes_tree_operand: PyNodeAttributesTreeOperand
    ) -> NodeAttributesTreeOperand:
        attributes_tree_operand = cls()
        attributes_tree_operand._attributes_tree_operand = py_attributes_tree_operand
        return attributes_tree_operand


class NodeAttributesTreeGroupOperand:
    _attributes_tree_operand: PyNodeAttributesTreeGroupOperand

    def max(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.max()
        )

    def min(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.min()
        )

    def count(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.count()
        )

    def sum(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.sum()
        )

    def random(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.random()
        )

    def is_string(self) -> None:
        self._attributes_tree_operand.is_string()

    def is_int(self) -> None:
        self._attributes_tree_operand.is_int()

    def is_max(self) -> None:
        self._attributes_tree_operand.is_max()

    def is_min(self) -> None:
        self._attributes_tree_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._attributes_tree_operand.abs()

    def trim(self) -> None:
        self._attributes_tree_operand.trim()

    def trim_start(self) -> None:
        self._attributes_tree_operand.trim_start()

    def trim_end(self) -> None:
        self._attributes_tree_operand.trim_end()

    def lowercase(self) -> None:
        self._attributes_tree_operand.lowercase()

    def uppercase(self) -> None:
        self._attributes_tree_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._attributes_tree_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeAttributesTreeOperand], None],
        or_: Callable[[NodeAttributesTreeOperand], None],
    ) -> None:
        self._attributes_tree_operand.either_or(
            lambda attributes: either(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
            lambda attributes: or_(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
        )

    def exclude(self, query: Callable[[NodeAttributesTreeOperand], None]) -> None:
        self._attributes_tree_operand.exclude(
            lambda attributes: query(
                NodeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            )
        )

    def ungroup(self) -> NodeAttributesTreeOperand:
        return NodeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.ungroup()
        )

    def clone(self) -> NodeAttributesTreeGroupOperand:
        return NodeAttributesTreeGroupOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.deep_clone()
        )

    @classmethod
    def _from_py_attributes_tree_operand(
        cls, py_attributes_tree_operand: PyNodeAttributesTreeGroupOperand
    ) -> NodeAttributesTreeGroupOperand:
        attributes_tree_operand = cls()
        attributes_tree_operand._attributes_tree_operand = py_attributes_tree_operand
        return attributes_tree_operand


class EdgeAttributesTreeOperand:
    _attributes_tree_operand: PyEdgeAttributesTreeOperand

    def max(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.max()
            )
        )

    def min(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.min()
            )
        )

    def count(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.count()
            )
        )

    def sum(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.sum()
            )
        )

    def random(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._attributes_tree_operand.random()
            )
        )

    def is_string(self) -> None:
        self._attributes_tree_operand.is_string()

    def is_int(self) -> None:
        self._attributes_tree_operand.is_int()

    def is_max(self) -> None:
        self._attributes_tree_operand.is_max()

    def is_min(self) -> None:
        self._attributes_tree_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._attributes_tree_operand.abs()

    def trim(self) -> None:
        self._attributes_tree_operand.trim()

    def trim_start(self) -> None:
        self._attributes_tree_operand.trim_start()

    def trim_end(self) -> None:
        self._attributes_tree_operand.trim_end()

    def lowercase(self) -> None:
        self._attributes_tree_operand.lowercase()

    def uppercase(self) -> None:
        self._attributes_tree_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._attributes_tree_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeAttributesTreeOperand], None],
        or_: Callable[[EdgeAttributesTreeOperand], None],
    ) -> None:
        self._attributes_tree_operand.either_or(
            lambda attributes: either(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
            lambda attributes: or_(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
        )

    def exclude(self, query: Callable[[EdgeAttributesTreeOperand], None]) -> None:
        self._attributes_tree_operand.exclude(
            lambda attributes: query(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            )
        )

    def clone(self) -> EdgeAttributesTreeOperand:
        return EdgeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.deep_clone()
        )

    @classmethod
    def _from_py_attributes_tree_operand(
        cls, py_attributes_tree_operand: PyEdgeAttributesTreeOperand
    ) -> EdgeAttributesTreeOperand:
        attributes_tree_operand = cls()
        attributes_tree_operand._attributes_tree_operand = py_attributes_tree_operand
        return attributes_tree_operand


class EdgeAttributesTreeGroupOperand:
    _attributes_tree_operand: PyEdgeAttributesTreeGroupOperand

    def max(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.max()
        )

    def min(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.min()
        )

    def count(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.count()
        )

    def sum(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.sum()
        )

    def random(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.random()
        )

    def is_string(self) -> None:
        self._attributes_tree_operand.is_string()

    def is_int(self) -> None:
        self._attributes_tree_operand.is_int()

    def is_max(self) -> None:
        self._attributes_tree_operand.is_max()

    def is_min(self) -> None:
        self._attributes_tree_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._attributes_tree_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._attributes_tree_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._attributes_tree_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._attributes_tree_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._attributes_tree_operand.abs()

    def trim(self) -> None:
        self._attributes_tree_operand.trim()

    def trim_start(self) -> None:
        self._attributes_tree_operand.trim_start()

    def trim_end(self) -> None:
        self._attributes_tree_operand.trim_end()

    def lowercase(self) -> None:
        self._attributes_tree_operand.lowercase()

    def uppercase(self) -> None:
        self._attributes_tree_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._attributes_tree_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeAttributesTreeOperand], None],
        or_: Callable[[EdgeAttributesTreeOperand], None],
    ) -> None:
        self._attributes_tree_operand.either_or(
            lambda attributes: either(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
            lambda attributes: or_(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
        )

    def exclude(self, query: Callable[[EdgeAttributesTreeOperand], None]) -> None:
        self._attributes_tree_operand.exclude(
            lambda attributes: query(
                EdgeAttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            )
        )

    def ungroup(self) -> EdgeAttributesTreeOperand:
        return EdgeAttributesTreeOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.ungroup()
        )

    def clone(self) -> EdgeAttributesTreeGroupOperand:
        return EdgeAttributesTreeGroupOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.deep_clone()
        )

    @classmethod
    def _from_py_attributes_tree_operand(
        cls, py_attributes_tree_operand: PyEdgeAttributesTreeGroupOperand
    ) -> EdgeAttributesTreeGroupOperand:
        attributes_tree_operand = cls()
        attributes_tree_operand._attributes_tree_operand = py_attributes_tree_operand
        return attributes_tree_operand


class NodeMultipleAttributesWithIndexOperand:
    _multiple_attributes_operand: PyNodeMultipleAttributesWithIndexOperand

    def max(self) -> NodeSingleAttributeWithIndexOperand:
        return NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.max()
        )

    def min(self) -> NodeSingleAttributeWithIndexOperand:
        return NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.min()
        )

    def count(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> NodeSingleAttributeWithIndexOperand:
        return NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def to_values(self) -> NodeMultipleValuesWithIndexOperand:
        return NodeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_attributes_operand.to_values()
        )

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleAttributesWithIndexOperand], None],
        or_: Callable[[NodeMultipleAttributesWithIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleAttributesWithIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def clone(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._multiple_attributes_operand.deep_clone()
            )
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls, py_multiple_attributes_operand: PyNodeMultipleAttributesWithIndexOperand
    ) -> NodeMultipleAttributesWithIndexOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class NodeMultipleAttributesWithIndexGroupOperand:
    _multiple_attributes_operand: PyNodeMultipleAttributesWithIndexGroupOperand

    def max(self) -> NodeSingleAttributeWithIndexGroupOperand:
        return (
            NodeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.max()
            )
        )

    def min(self) -> NodeSingleAttributeWithIndexGroupOperand:
        return (
            NodeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.min()
            )
        )

    def count(self) -> NodeSingleAttributeWithoutIndexGroupOperand:
        return NodeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> NodeSingleAttributeWithoutIndexGroupOperand:
        return NodeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> NodeSingleAttributeWithIndexGroupOperand:
        return (
            NodeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.random()
            )
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def to_values(self) -> NodeMultipleValuesWithIndexGroupOperand:
        return NodeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._multiple_attributes_operand.to_values()
        )

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleAttributesWithIndexOperand], None],
        or_: Callable[[NodeMultipleAttributesWithIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleAttributesWithIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def ungroup(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._multiple_attributes_operand.ungroup()
            )
        )

    def clone(self) -> NodeMultipleAttributesWithIndexGroupOperand:
        return NodeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._multiple_attributes_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls,
        py_multiple_attributes_operand: PyNodeMultipleAttributesWithIndexGroupOperand,
    ) -> NodeMultipleAttributesWithIndexGroupOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class NodeMultipleAttributesWithoutIndexOperand:
    _multiple_attributes_operand: PyNodeMultipleAttributesWithoutIndexOperand

    def max(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.max()
        )

    def min(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.min()
        )

    def count(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeMultipleAttributesWithoutIndexOperand], None],
        or_: Callable[[NodeMultipleAttributesWithoutIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                NodeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                NodeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeMultipleAttributesWithoutIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                NodeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def clone(self) -> NodeMultipleAttributesWithoutIndexOperand:
        return NodeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
            self._multiple_attributes_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls, py_multiple_attributes_operand: PyNodeMultipleAttributesWithoutIndexOperand
    ) -> NodeMultipleAttributesWithoutIndexOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class EdgeMultipleAttributesWithIndexOperand:
    _multiple_attributes_operand: PyEdgeMultipleAttributesWithIndexOperand

    def max(self) -> EdgeSingleAttributeWithIndexOperand:
        return EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.max()
        )

    def min(self) -> EdgeSingleAttributeWithIndexOperand:
        return EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.min()
        )

    def count(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> EdgeSingleAttributeWithIndexOperand:
        return EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def to_values(self) -> EdgeMultipleValuesWithIndexOperand:
        return EdgeMultipleValuesWithIndexOperand._from_py_multiple_values_operand(
            self._multiple_attributes_operand.to_values()
        )

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleAttributesWithIndexOperand], None],
        or_: Callable[[EdgeMultipleAttributesWithIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleAttributesWithIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def clone(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._multiple_attributes_operand.deep_clone()
            )
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls, py_multiple_attributes_operand: PyEdgeMultipleAttributesWithIndexOperand
    ) -> EdgeMultipleAttributesWithIndexOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class EdgeMultipleAttributesWithIndexGroupOperand:
    _multiple_attributes_operand: PyEdgeMultipleAttributesWithIndexGroupOperand

    def max(self) -> EdgeSingleAttributeWithIndexGroupOperand:
        return (
            EdgeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.max()
            )
        )

    def min(self) -> EdgeSingleAttributeWithIndexGroupOperand:
        return (
            EdgeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.min()
            )
        )

    def count(self) -> EdgeSingleAttributeWithoutIndexGroupOperand:
        return EdgeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> EdgeSingleAttributeWithoutIndexGroupOperand:
        return EdgeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> EdgeSingleAttributeWithIndexGroupOperand:
        return (
            EdgeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._multiple_attributes_operand.random()
            )
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def to_values(self) -> EdgeMultipleValuesWithIndexGroupOperand:
        return EdgeMultipleValuesWithIndexGroupOperand._from_py_multiple_values_operand(
            self._multiple_attributes_operand.to_values()
        )

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleAttributesWithIndexOperand], None],
        or_: Callable[[EdgeMultipleAttributesWithIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleAttributesWithIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def ungroup(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._multiple_attributes_operand.ungroup()
            )
        )

    def clone(self) -> EdgeMultipleAttributesWithIndexGroupOperand:
        return EdgeMultipleAttributesWithIndexGroupOperand._from_py_multiple_attributes_operand(
            self._multiple_attributes_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls,
        py_multiple_attributes_operand: PyEdgeMultipleAttributesWithIndexGroupOperand,
    ) -> EdgeMultipleAttributesWithIndexGroupOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class EdgeMultipleAttributesWithoutIndexOperand:
    _multiple_attributes_operand: PyEdgeMultipleAttributesWithoutIndexOperand

    def max(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.max()
        )

    def min(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.min()
        )

    def count(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def random(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.random()
        )

    def is_string(self) -> None:
        self._multiple_attributes_operand.is_string()

    def is_int(self) -> None:
        self._multiple_attributes_operand.is_int()

    def is_max(self) -> None:
        self._multiple_attributes_operand.is_max()

    def is_min(self) -> None:
        self._multiple_attributes_operand.is_min()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._multiple_attributes_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._multiple_attributes_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._multiple_attributes_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._multiple_attributes_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._multiple_attributes_operand.abs()

    def trim(self) -> None:
        self._multiple_attributes_operand.trim()

    def trim_start(self) -> None:
        self._multiple_attributes_operand.trim_start()

    def trim_end(self) -> None:
        self._multiple_attributes_operand.trim_end()

    def lowercase(self) -> None:
        self._multiple_attributes_operand.lowercase()

    def uppercase(self) -> None:
        self._multiple_attributes_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeMultipleAttributesWithoutIndexOperand], None],
        or_: Callable[[EdgeMultipleAttributesWithoutIndexOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                EdgeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                EdgeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeMultipleAttributesWithoutIndexOperand], None]
    ) -> None:
        self._multiple_attributes_operand.exclude(
            lambda attributes: query(
                EdgeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            )
        )

    def clone(self) -> EdgeMultipleAttributesWithoutIndexOperand:
        return EdgeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
            self._multiple_attributes_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls, py_multiple_attributes_operand: PyEdgeMultipleAttributesWithoutIndexOperand
    ) -> EdgeMultipleAttributesWithoutIndexOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class NodeSingleAttributeWithIndexOperand:
    _single_attribute_operand: PyNodeSingleAttributeWithIndexOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleAttributeWithIndexOperand], None],
        or_: Callable[[NodeSingleAttributeWithIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleAttributeWithIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def clone(self) -> NodeSingleAttributeWithIndexOperand:
        return NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyNodeSingleAttributeWithIndexOperand
    ) -> NodeSingleAttributeWithIndexOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class NodeSingleAttributeWithIndexGroupOperand:
    _single_attribute_operand: PyNodeSingleAttributeWithIndexGroupOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleAttributeWithIndexOperand], None],
        or_: Callable[[NodeSingleAttributeWithIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleAttributeWithIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                NodeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def ungroup(self) -> NodeMultipleAttributesWithIndexOperand:
        return (
            NodeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._single_attribute_operand.ungroup()
            )
        )

    def clone(self) -> NodeSingleAttributeWithIndexGroupOperand:
        return (
            NodeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._single_attribute_operand.deep_clone()
            )
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyNodeSingleAttributeWithIndexGroupOperand
    ) -> NodeSingleAttributeWithIndexGroupOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class NodeSingleAttributeWithoutIndexOperand:
    _single_attribute_operand: PyNodeSingleAttributeWithoutIndexOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleAttributeWithoutIndexOperand], None],
        or_: Callable[[NodeSingleAttributeWithoutIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleAttributeWithoutIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def clone(self) -> NodeSingleAttributeWithoutIndexOperand:
        return NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyNodeSingleAttributeWithoutIndexOperand
    ) -> NodeSingleAttributeWithoutIndexOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class NodeSingleAttributeWithoutIndexGroupOperand:
    _single_attribute_operand: PyNodeSingleAttributeWithoutIndexGroupOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeSingleAttributeWithoutIndexOperand], None],
        or_: Callable[[NodeSingleAttributeWithoutIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[NodeSingleAttributeWithoutIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                NodeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def ungroup(self) -> NodeMultipleAttributesWithoutIndexOperand:
        return NodeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
            self._single_attribute_operand.ungroup()
        )

    def clone(self) -> NodeSingleAttributeWithoutIndexGroupOperand:
        return NodeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyNodeSingleAttributeWithoutIndexGroupOperand
    ) -> NodeSingleAttributeWithoutIndexGroupOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class EdgeSingleAttributeWithIndexOperand:
    _single_attribute_operand: PyEdgeSingleAttributeWithIndexOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleAttributeWithIndexOperand], None],
        or_: Callable[[EdgeSingleAttributeWithIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleAttributeWithIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def clone(self) -> EdgeSingleAttributeWithIndexOperand:
        return EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyEdgeSingleAttributeWithIndexOperand
    ) -> EdgeSingleAttributeWithIndexOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class EdgeSingleAttributeWithIndexGroupOperand:
    _single_attribute_operand: PyEdgeSingleAttributeWithIndexGroupOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleAttributeWithIndexOperand], None],
        or_: Callable[[EdgeSingleAttributeWithIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleAttributeWithIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                EdgeSingleAttributeWithIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def ungroup(self) -> EdgeMultipleAttributesWithIndexOperand:
        return (
            EdgeMultipleAttributesWithIndexOperand._from_py_multiple_attributes_operand(
                self._single_attribute_operand.ungroup()
            )
        )

    def clone(self) -> EdgeSingleAttributeWithIndexGroupOperand:
        return (
            EdgeSingleAttributeWithIndexGroupOperand._from_py_single_attribute_operand(
                self._single_attribute_operand.deep_clone()
            )
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyEdgeSingleAttributeWithIndexGroupOperand
    ) -> EdgeSingleAttributeWithIndexGroupOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class EdgeSingleAttributeWithoutIndexOperand:
    _single_attribute_operand: PyEdgeSingleAttributeWithoutIndexOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleAttributeWithoutIndexOperand], None],
        or_: Callable[[EdgeSingleAttributeWithoutIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleAttributeWithoutIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def clone(self) -> EdgeSingleAttributeWithoutIndexOperand:
        return EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyEdgeSingleAttributeWithoutIndexOperand
    ) -> EdgeSingleAttributeWithoutIndexOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class EdgeSingleAttributeWithoutIndexGroupOperand:
    _single_attribute_operand: PyEdgeSingleAttributeWithoutIndexGroupOperand

    def is_string(self) -> None:
        self._single_attribute_operand.is_string()

    def is_int(self) -> None:
        self._single_attribute_operand.is_int()

    def greater_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.greater_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def greater_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.greater_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.less_than(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def less_than_or_equal_to(
        self, attribute: SingleAttributeComparisonOperand
    ) -> None:
        self._single_attribute_operand.less_than_or_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def not_equal_to(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.not_equal_to(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def is_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def is_not_in(self, attributes: MultipleAttributesComparisonOperand) -> None:
        self._single_attribute_operand.is_not_in(
            _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
                attributes
            )
        )

    def starts_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.starts_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def ends_with(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.ends_with(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def contains(self, attribute: SingleAttributeComparisonOperand) -> None:
        self._single_attribute_operand.contains(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def add(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.add(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def subtract(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.sub(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def multiply(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mul(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def modulo(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.mod(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def power(self, attribute: SingleAttributeArithmeticOperand) -> None:
        self._single_attribute_operand.pow(
            _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
                attribute
            )
        )

    def absolute(self) -> None:
        self._single_attribute_operand.abs()

    def trim(self) -> None:
        self._single_attribute_operand.trim()

    def trim_start(self) -> None:
        self._single_attribute_operand.trim_start()

    def trim_end(self) -> None:
        self._single_attribute_operand.trim_end()

    def lowercase(self) -> None:
        self._single_attribute_operand.lowercase()

    def uppercase(self) -> None:
        self._single_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._single_attribute_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[EdgeSingleAttributeWithoutIndexOperand], None],
        or_: Callable[[EdgeSingleAttributeWithoutIndexOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
            lambda attribute: or_(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            ),
        )

    def exclude(
        self, query: Callable[[EdgeSingleAttributeWithoutIndexOperand], None]
    ) -> None:
        self._single_attribute_operand.exclude(
            lambda attribute: query(
                EdgeSingleAttributeWithoutIndexOperand._from_py_single_attribute_operand(
                    attribute
                )
            )
        )

    def ungroup(self) -> EdgeMultipleAttributesWithoutIndexOperand:
        return EdgeMultipleAttributesWithoutIndexOperand._from_py_multiple_attributes_operand(
            self._single_attribute_operand.ungroup()
        )

    def clone(self) -> EdgeSingleAttributeWithoutIndexGroupOperand:
        return EdgeSingleAttributeWithoutIndexGroupOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PyEdgeSingleAttributeWithoutIndexGroupOperand
    ) -> EdgeSingleAttributeWithoutIndexGroupOperand:
        single_attribute_operand = cls()
        single_attribute_operand._single_attribute_operand = py_single_attribute_operand
        return single_attribute_operand


class NodeIndicesOperand:
    _node_indices_operand: PyNodeIndicesOperand

    def max(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.max()
        )

    def min(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.min()
        )

    def count(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.count()
        )

    def sum(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.sum()
        )

    def random(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.random()
        )

    def is_string(self) -> None:
        self._node_indices_operand.is_string()

    def is_int(self) -> None:
        self._node_indices_operand.is_int()

    def is_max(self) -> None:
        self._node_indices_operand.is_max()

    def is_min(self) -> None:
        self._node_indices_operand.is_min()

    def greater_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.greater_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.greater_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.less_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.less_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def not_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.not_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def is_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_indices_operand.is_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_indices_operand.is_not_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.starts_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def ends_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.ends_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def contains(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.contains(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def add(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.add(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def subtract(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.sub(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def multiply(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.mul(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def modulo(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.mod(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def power(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.pow(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def absolute(self) -> None:
        self._node_indices_operand.abs()

    def trim(self) -> None:
        self._node_indices_operand.trim()

    def trim_start(self) -> None:
        self._node_indices_operand.trim_start()

    def trim_end(self) -> None:
        self._node_indices_operand.trim_end()

    def lowercase(self) -> None:
        self._node_indices_operand.lowercase()

    def uppercase(self) -> None:
        self._node_indices_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._node_indices_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeIndicesOperand], None],
        or_: Callable[[NodeIndicesOperand], None],
    ) -> None:
        self._node_indices_operand.either_or(
            lambda node_indices: either(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            ),
            lambda node_indices: or_(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            ),
        )

    def exclude(self, query: Callable[[NodeIndicesOperand], None]) -> None:
        self._node_indices_operand.exclude(
            lambda node_indices: query(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            )
        )

    def clone(self) -> NodeIndicesOperand:
        return NodeIndicesOperand._from_py_node_indices_operand(
            self._node_indices_operand.deep_clone()
        )

    @classmethod
    def _from_py_node_indices_operand(
        cls, py_node_indices_operand: PyNodeIndicesOperand
    ) -> NodeIndicesOperand:
        node_indices_operand = cls()
        node_indices_operand._node_indices_operand = py_node_indices_operand
        return node_indices_operand


class NodeIndicesGroupOperand:
    _node_indices_operand: PyNodeIndicesGroupOperand

    def max(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_indices_operand.max()
        )

    def min(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_indices_operand.min()
        )

    def count(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_indices_operand.count()
        )

    def sum(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_indices_operand.sum()
        )

    def random(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_indices_operand.random()
        )

    def is_string(self) -> None:
        self._node_indices_operand.is_string()

    def is_int(self) -> None:
        self._node_indices_operand.is_int()

    def is_max(self) -> None:
        self._node_indices_operand.is_max()

    def is_min(self) -> None:
        self._node_indices_operand.is_min()

    def greater_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.greater_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.greater_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.less_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.less_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def not_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.not_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def is_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_indices_operand.is_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_indices_operand.is_not_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.starts_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def ends_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.ends_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def contains(self, index: NodeIndexComparisonOperand) -> None:
        self._node_indices_operand.contains(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def add(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.add(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def subtract(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.sub(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def multiply(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.mul(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def modulo(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.mod(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def power(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_indices_operand.pow(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def absolute(self) -> None:
        self._node_indices_operand.abs()

    def trim(self) -> None:
        self._node_indices_operand.trim()

    def trim_start(self) -> None:
        self._node_indices_operand.trim_start()

    def trim_end(self) -> None:
        self._node_indices_operand.trim_end()

    def lowercase(self) -> None:
        self._node_indices_operand.lowercase()

    def uppercase(self) -> None:
        self._node_indices_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._node_indices_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeIndicesOperand], None],
        or_: Callable[[NodeIndicesOperand], None],
    ) -> None:
        self._node_indices_operand.either_or(
            lambda node_indices: either(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            ),
            lambda node_indices: or_(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            ),
        )

    def exclude(self, query: Callable[[NodeIndicesOperand], None]) -> None:
        self._node_indices_operand.exclude(
            lambda node_indices: query(
                NodeIndicesOperand._from_py_node_indices_operand(node_indices)
            )
        )

    def ungroup(self) -> NodeIndicesOperand:
        return NodeIndicesOperand._from_py_node_indices_operand(
            self._node_indices_operand.ungroup()
        )

    def clone(self) -> NodeIndicesGroupOperand:
        return NodeIndicesGroupOperand._from_py_node_indices_operand(
            self._node_indices_operand.deep_clone()
        )

    @classmethod
    def _from_py_node_indices_operand(
        cls, py_node_indices_operand: PyNodeIndicesGroupOperand
    ) -> NodeIndicesGroupOperand:
        node_indices_operand = cls()
        node_indices_operand._node_indices_operand = py_node_indices_operand
        return node_indices_operand


class NodeIndexOperand:
    _node_index_operand: PyNodeIndexOperand

    def is_string(self) -> None:
        self._node_index_operand.is_string()

    def is_int(self) -> None:
        self._node_index_operand.is_int()

    def greater_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.greater_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.greater_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.less_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.less_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def not_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.not_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def is_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_index_operand.is_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_index_operand.is_not_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.starts_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def ends_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.ends_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def contains(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.contains(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def add(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.add(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def subtract(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.sub(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def multiply(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.mul(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def modulo(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.mod(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def power(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.pow(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def absolute(self) -> None:
        self._node_index_operand.abs()

    def trim(self) -> None:
        self._node_index_operand.trim()

    def trim_start(self) -> None:
        self._node_index_operand.trim_start()

    def trim_end(self) -> None:
        self._node_index_operand.trim_end()

    def lowercase(self) -> None:
        self._node_index_operand.lowercase()

    def uppercase(self) -> None:
        self._node_index_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._node_index_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeIndexOperand], None],
        or_: Callable[[NodeIndexOperand], None],
    ) -> None:
        self._node_index_operand.either_or(
            lambda node_index: either(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            ),
            lambda node_index: or_(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            ),
        )

    def exclude(self, query: Callable[[NodeIndexOperand], None]) -> None:
        self._node_index_operand.exclude(
            lambda node_index: query(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            )
        )

    def clone(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_index_operand.deep_clone()
        )

    @classmethod
    def _from_py_node_index_operand(
        cls, py_node_index_operand: PyNodeIndexOperand
    ) -> NodeIndexOperand:
        node_index_operand = cls()
        node_index_operand._node_index_operand = py_node_index_operand
        return node_index_operand


class NodeIndexGroupOperand:
    _node_index_operand: PyNodeIndexGroupOperand

    def is_string(self) -> None:
        self._node_index_operand.is_string()

    def is_int(self) -> None:
        self._node_index_operand.is_int()

    def greater_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.greater_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.greater_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.less_than(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.less_than_or_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def not_equal_to(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.not_equal_to(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def is_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_index_operand.is_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: NodeIndicesComparisonOperand) -> None:
        self._node_index_operand.is_not_in(
            _py_node_indices_comparison_operand_from_node_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.starts_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def ends_with(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.ends_with(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def contains(self, index: NodeIndexComparisonOperand) -> None:
        self._node_index_operand.contains(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def add(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.add(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def subtract(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.sub(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def multiply(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.mul(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def modulo(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.mod(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def power(self, index: NodeIndexArithmeticOperand) -> None:
        self._node_index_operand.pow(
            _py_node_index_comparison_operand_from_node_index_comparison_operand(index)
        )

    def absolute(self) -> None:
        self._node_index_operand.abs()

    def trim(self) -> None:
        self._node_index_operand.trim()

    def trim_start(self) -> None:
        self._node_index_operand.trim_start()

    def trim_end(self) -> None:
        self._node_index_operand.trim_end()

    def lowercase(self) -> None:
        self._node_index_operand.lowercase()

    def uppercase(self) -> None:
        self._node_index_operand.uppercase()

    def slice(self, start: int, end: int) -> None:
        self._node_index_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[NodeIndexOperand], None],
        or_: Callable[[NodeIndexOperand], None],
    ) -> None:
        self._node_index_operand.either_or(
            lambda node_index: either(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            ),
            lambda node_index: or_(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            ),
        )

    def exclude(self, query: Callable[[NodeIndexOperand], None]) -> None:
        self._node_index_operand.exclude(
            lambda node_index: query(
                NodeIndexOperand._from_py_node_index_operand(node_index)
            )
        )

    def ungroup(self) -> NodeIndicesOperand:
        return NodeIndicesOperand._from_py_node_indices_operand(
            self._node_index_operand.ungroup()
        )

    def clone(self) -> NodeIndexGroupOperand:
        return NodeIndexGroupOperand._from_py_node_index_operand(
            self._node_index_operand.deep_clone()
        )

    @classmethod
    def _from_py_node_index_operand(
        cls, py_node_index_operand: PyNodeIndexGroupOperand
    ) -> NodeIndexGroupOperand:
        node_index_operand = cls()
        node_index_operand._node_index_operand = py_node_index_operand
        return node_index_operand


class EdgeIndicesOperand:
    _edge_indices_operand: PyEdgeIndicesOperand

    def max(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.max()
        )

    def min(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.min()
        )

    def count(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.count()
        )

    def sum(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.sum()
        )

    def random(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.random()
        )

    def is_max(self) -> None:
        self._edge_indices_operand.is_max()

    def is_min(self) -> None:
        self._edge_indices_operand.is_min()

    def greater_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.greater_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.greater_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.less_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.less_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def not_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.not_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def is_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_indices_operand.is_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_indices_operand.is_not_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.starts_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def ends_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.ends_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def contains(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.contains(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def add(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.add(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def subtract(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.sub(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def multiply(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.mul(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def modulo(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.mod(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def power(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.pow(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def either_or(
        self,
        either: Callable[[EdgeIndicesOperand], None],
        or_: Callable[[EdgeIndicesOperand], None],
    ) -> None:
        self._edge_indices_operand.either_or(
            lambda edge_indices: either(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            ),
            lambda edge_indices: or_(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            ),
        )

    def exclude(self, query: Callable[[EdgeIndicesOperand], None]) -> None:
        self._edge_indices_operand.exclude(
            lambda edge_indices: query(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            )
        )

    def clone(self) -> EdgeIndicesOperand:
        return EdgeIndicesOperand._from_edge_indices_operand(
            self._edge_indices_operand.deep_clone()
        )

    @classmethod
    def _from_edge_indices_operand(
        cls, py_edge_indices_operand: PyEdgeIndicesOperand
    ) -> EdgeIndicesOperand:
        edge_indices_operand = cls()
        edge_indices_operand._edge_indices_operand = py_edge_indices_operand
        return edge_indices_operand


class EdgeIndicesGroupOperand:
    _edge_indices_operand: PyEdgeIndicesGroupOperand

    def max(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_indices_operand.max()
        )

    def min(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_indices_operand.min()
        )

    def count(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_indices_operand.count()
        )

    def sum(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_indices_operand.sum()
        )

    def random(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_indices_operand.random()
        )

    def is_max(self) -> None:
        self._edge_indices_operand.is_max()

    def is_min(self) -> None:
        self._edge_indices_operand.is_min()

    def greater_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.greater_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.greater_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.less_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.less_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def not_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.not_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def is_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_indices_operand.is_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_indices_operand.is_not_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.starts_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def ends_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.ends_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def contains(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_indices_operand.contains(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def add(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.add(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def subtract(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.sub(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def multiply(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.mul(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def modulo(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.mod(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def power(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_indices_operand.pow(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def either_or(
        self,
        either: Callable[[EdgeIndicesOperand], None],
        or_: Callable[[EdgeIndicesOperand], None],
    ) -> None:
        self._edge_indices_operand.either_or(
            lambda edge_indices: either(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            ),
            lambda edge_indices: or_(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            ),
        )

    def exclude(self, query: Callable[[EdgeIndicesOperand], None]) -> None:
        self._edge_indices_operand.exclude(
            lambda edge_indices: query(
                EdgeIndicesOperand._from_edge_indices_operand(edge_indices)
            )
        )

    def ungroup(self) -> EdgeIndicesOperand:
        return EdgeIndicesOperand._from_edge_indices_operand(
            self._edge_indices_operand.ungroup()
        )

    def clone(self) -> EdgeIndicesGroupOperand:
        return EdgeIndicesGroupOperand._from_edge_indices_operand(
            self._edge_indices_operand.deep_clone()
        )

    @classmethod
    def _from_edge_indices_operand(
        cls, py_edge_indices_operand: PyEdgeIndicesGroupOperand
    ) -> EdgeIndicesGroupOperand:
        edge_indices_operand = cls()
        edge_indices_operand._edge_indices_operand = py_edge_indices_operand
        return edge_indices_operand


class EdgeIndexOperand:
    _edge_index_operand: PyEdgeIndexOperand

    def greater_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.greater_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.greater_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.less_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.less_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def not_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.not_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def is_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_index_operand.is_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_index_operand.is_not_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.starts_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def ends_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.ends_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def contains(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.contains(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def add(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.add(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def subtract(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.sub(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def multiply(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.mul(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def modulo(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.mod(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def power(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.pow(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def either_or(
        self,
        either: Callable[[EdgeIndexOperand], None],
        or_: Callable[[EdgeIndexOperand], None],
    ) -> None:
        self._edge_index_operand.either_or(
            lambda edge_index: either(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            ),
            lambda edge_index: or_(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            ),
        )

    def exclude(self, query: Callable[[EdgeIndexOperand], None]) -> None:
        self._edge_index_operand.exclude(
            lambda edge_index: query(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            )
        )

    def clone(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_index_operand.deep_clone()
        )

    @classmethod
    def _from_py_edge_index_operand(
        cls, py_edge_index_operand: PyEdgeIndexOperand
    ) -> EdgeIndexOperand:
        edge_index_operand = cls()
        edge_index_operand._edge_index_operand = py_edge_index_operand
        return edge_index_operand


class EdgeIndexGroupOperand:
    _edge_index_operand: PyEdgeIndexGroupOperand

    def greater_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.greater_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def greater_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.greater_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.less_than(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def less_than_or_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.less_than_or_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def not_equal_to(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.not_equal_to(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def is_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_index_operand.is_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def is_not_in(self, indices: EdgeIndicesComparisonOperand) -> None:
        self._edge_index_operand.is_not_in(
            _py_edge_indices_comparison_operand_from_edge_indices_comparison_operand(
                indices
            )
        )

    def starts_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.starts_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def ends_with(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.ends_with(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def contains(self, index: EdgeIndexComparisonOperand) -> None:
        self._edge_index_operand.contains(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def add(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.add(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def subtract(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.sub(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def multiply(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.mul(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def modulo(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.mod(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def power(self, index: EdgeIndexArithmeticOperand) -> None:
        self._edge_index_operand.pow(
            _py_edge_index_comparison_operand_from_edge_index_comparison_operand(index)
        )

    def either_or(
        self,
        either: Callable[[EdgeIndexOperand], None],
        or_: Callable[[EdgeIndexOperand], None],
    ) -> None:
        self._edge_index_operand.either_or(
            lambda edge_index: either(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            ),
            lambda edge_index: or_(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            ),
        )

    def exclude(self, query: Callable[[EdgeIndexOperand], None]) -> None:
        self._edge_index_operand.exclude(
            lambda edge_index: query(
                EdgeIndexOperand._from_py_edge_index_operand(edge_index)
            )
        )

    def ungroup(self) -> EdgeIndicesOperand:
        return EdgeIndicesOperand._from_edge_indices_operand(
            self._edge_index_operand.ungroup()
        )

    def clone(self) -> EdgeIndexGroupOperand:
        return EdgeIndexGroupOperand._from_py_edge_index_operand(
            self._edge_index_operand.deep_clone()
        )

    @classmethod
    def _from_py_edge_index_operand(
        cls, py_edge_index_operand: PyEdgeIndexGroupOperand
    ) -> EdgeIndexGroupOperand:
        edge_index_operand = cls()
        edge_index_operand._edge_index_operand = py_edge_index_operand
        return edge_index_operand

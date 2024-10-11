from __future__ import annotations

import sys
from enum import Enum
from typing import Callable, List, Union

from medmodels._medmodels import (
    PyAttributesTreeOperand,
    PyEdgeDirection,
    PyEdgeIndexOperand,
    PyEdgeIndicesOperand,
    PyEdgeOperand,
    PyMultipleAttributesOperand,
    PyMultipleValuesOperand,
    PyNodeIndexOperand,
    PyNodeIndicesOperand,
    PyNodeOperand,
    PySingleAttributeOperand,
    PySingleValueOperand,
)
from medmodels.medrecord.types import (
    EdgeIndex,
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

NodeQuery: TypeAlias = Callable[["NodeOperand"], None]
EdgeQuery: TypeAlias = Callable[["EdgeOperand"], None]

SingleValueComparisonOperand: TypeAlias = Union["SingleValueOperand", MedRecordValue]
SingleValueArithmeticOperand: TypeAlias = SingleValueComparisonOperand
MultipleValuesComparisonOperand: TypeAlias = Union[
    "MultipleValuesOperand", List[MedRecordValue]
]


def _py_single_value_comparison_operand_from_single_value_comparison_operand(
    single_value_comparison_operand: SingleValueComparisonOperand,
) -> Union[MedRecordValue, PySingleValueOperand]:
    if isinstance(single_value_comparison_operand, SingleValueOperand):
        return single_value_comparison_operand._single_value_operand
    return single_value_comparison_operand


def _py_multiple_values_comparison_operand_from_multiple_values_comparison_operand(
    multiple_values_comparison_operand: MultipleValuesComparisonOperand,
) -> Union[List[MedRecordValue], PyMultipleValuesOperand]:
    if isinstance(multiple_values_comparison_operand, MultipleValuesOperand):
        return multiple_values_comparison_operand._multiple_values_operand
    return multiple_values_comparison_operand


SingleAttributeComparisonOperand: TypeAlias = Union[
    "SingleAttributeOperand",
    MedRecordAttribute,
]
SingleAttributeArithmeticOperand: TypeAlias = SingleAttributeComparisonOperand
MultipleAttributesComparisonOperand: TypeAlias = Union[
    "MultipleAttributesOperand", List[MedRecordAttribute]
]


def _py_single_attribute_comparison_operand_from_single_attribute_comparison_operand(
    single_attribute_comparison_operand: SingleAttributeComparisonOperand,
) -> Union[MedRecordAttribute, PySingleAttributeOperand]:
    if isinstance(single_attribute_comparison_operand, SingleAttributeOperand):
        return single_attribute_comparison_operand._single_attribute_operand
    return single_attribute_comparison_operand


def _py_multiple_attributes_comparison_operand_from_multiple_attributes_comparison_operand(
    multiple_attributes_comparison_operand: MultipleAttributesComparisonOperand,
) -> Union[List[MedRecordAttribute], PyMultipleAttributesOperand]:
    if isinstance(multiple_attributes_comparison_operand, MultipleAttributesOperand):
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
    INCOMING = 0
    OUTGOING = 1
    BOTH = 2


class NodeOperand:
    _node_operand: PyNodeOperand

    def attribute(self, attribute: MedRecordAttribute) -> MultipleValuesOperand:
        return MultipleValuesOperand._from_py_multiple_values_operand(
            self._node_operand.attribute(attribute)
        )

    def attributes(self) -> AttributesTreeOperand:
        return AttributesTreeOperand._from_py_attributes_tree_operand(
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

    def outgoing_edges(self) -> EdgeOperand:
        return EdgeOperand._from_py_edge_operand(self._node_operand.outgoing_edges())

    def incoming_edges(self) -> EdgeOperand:
        return EdgeOperand._from_py_edge_operand(self._node_operand.incoming_edges())

    def neighbors(
        self, edge_direction: EdgeDirection = EdgeDirection.OUTGOING
    ) -> NodeOperand:
        py_edge_direction = (
            PyEdgeDirection.Incoming
            if edge_direction == EdgeDirection.INCOMING
            else PyEdgeDirection.Outgoing
            if edge_direction == EdgeDirection.OUTGOING
            else PyEdgeDirection.Both
        )

        return NodeOperand._from_py_node_operand(
            self._node_operand.neighbors(py_edge_direction)
        )

    def either_or(self, either: NodeQuery, or_: NodeQuery) -> None:
        self._node_operand.either_or(
            lambda node: either(NodeOperand._from_py_node_operand(node)),
            lambda node: or_(NodeOperand._from_py_node_operand(node)),
        )

    def clone(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._node_operand.deep_clone())

    @classmethod
    def _from_py_node_operand(cls, py_node_operand: PyNodeOperand) -> NodeOperand:
        node_operand = cls()
        node_operand._node_operand = py_node_operand
        return node_operand


class EdgeOperand:
    _edge_operand: PyEdgeOperand

    def attribute(self, attribute: MedRecordAttribute) -> MultipleValuesOperand:
        return MultipleValuesOperand._from_py_multiple_values_operand(
            self._edge_operand.attribute(attribute)
        )

    def attributes(self) -> AttributesTreeOperand:
        return AttributesTreeOperand._from_py_attributes_tree_operand(
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

    def either_or(self, either: EdgeQuery, or_: EdgeQuery) -> None:
        self._edge_operand.either_or(
            lambda edge: either(EdgeOperand._from_py_edge_operand(edge)),
            lambda edge: or_(EdgeOperand._from_py_edge_operand(edge)),
        )

    def clone(self) -> EdgeOperand:
        return EdgeOperand._from_py_edge_operand(self._edge_operand.deep_clone())

    @classmethod
    def _from_py_edge_operand(cls, py_edge_operand: PyEdgeOperand) -> EdgeOperand:
        edge_operand = cls()
        edge_operand._edge_operand = py_edge_operand
        return edge_operand


class MultipleValuesOperand:
    _multiple_values_operand: PyMultipleValuesOperand

    def max(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.max()
        )

    def min(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.min()
        )

    def mean(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.mean()
        )

    def median(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.median()
        )

    def mode(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.mode()
        )

    def std(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.std()
        )

    def var(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.var()
        )

    def count(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.count()
        )

    def sum(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.sum()
        )

    def first(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.first()
        )

    def last(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._multiple_values_operand.last()
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
        either: Callable[[MultipleValuesOperand], None],
        or_: Callable[[MultipleValuesOperand], None],
    ) -> None:
        self._multiple_values_operand.either_or(
            lambda values: either(
                MultipleValuesOperand._from_py_multiple_values_operand(values)
            ),
            lambda values: or_(
                MultipleValuesOperand._from_py_multiple_values_operand(values)
            ),
        )

    def clone(self) -> MultipleValuesOperand:
        return MultipleValuesOperand._from_py_multiple_values_operand(
            self._multiple_values_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_values_operand(
        cls, py_multiple_values_operand: PyMultipleValuesOperand
    ) -> MultipleValuesOperand:
        multiple_values_operand = cls()
        multiple_values_operand._multiple_values_operand = py_multiple_values_operand
        return multiple_values_operand


class SingleValueOperand:
    _single_value_operand: PySingleValueOperand

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
        either: Callable[[SingleValueOperand], None],
        or_: Callable[[SingleValueOperand], None],
    ) -> None:
        self._single_value_operand.either_or(
            lambda value: either(
                SingleValueOperand._from_py_single_value_operand(value)
            ),
            lambda value: or_(SingleValueOperand._from_py_single_value_operand(value)),
        )

    def clone(self) -> SingleValueOperand:
        return SingleValueOperand._from_py_single_value_operand(
            self._single_value_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_value_operand(
        cls, py_single_value_operand: PySingleValueOperand
    ) -> SingleValueOperand:
        single_value_operand = cls()
        single_value_operand._single_value_operand = py_single_value_operand
        return single_value_operand


class AttributesTreeOperand:
    _attributes_tree_operand: PyAttributesTreeOperand

    def max(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.max()
        )

    def min(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.min()
        )

    def count(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.count()
        )

    def sum(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.sum()
        )

    def first(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.first()
        )

    def last(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._attributes_tree_operand.last()
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
        either: Callable[[AttributesTreeOperand], None],
        or_: Callable[[AttributesTreeOperand], None],
    ) -> None:
        self._attributes_tree_operand.either_or(
            lambda attributes: either(
                AttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
            lambda attributes: or_(
                AttributesTreeOperand._from_py_attributes_tree_operand(attributes)
            ),
        )

    def clone(self) -> AttributesTreeOperand:
        return AttributesTreeOperand._from_py_attributes_tree_operand(
            self._attributes_tree_operand.deep_clone()
        )

    @classmethod
    def _from_py_attributes_tree_operand(
        cls, py_attributes_tree_operand: PyAttributesTreeOperand
    ) -> AttributesTreeOperand:
        attributes_tree_operand = cls()
        attributes_tree_operand._attributes_tree_operand = py_attributes_tree_operand
        return attributes_tree_operand


class MultipleAttributesOperand:
    _multiple_attributes_operand: PyMultipleAttributesOperand

    def max(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.max()
        )

    def min(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.min()
        )

    def count(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.count()
        )

    def sum(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.sum()
        )

    def first(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.first()
        )

    def last(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._multiple_attributes_operand.last()
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

    def to_values(self) -> MultipleValuesOperand:
        return MultipleValuesOperand._from_py_multiple_values_operand(
            self._multiple_attributes_operand.to_values()
        )

    def slice(self, start: int, end: int) -> None:
        self._multiple_attributes_operand.slice(start, end)

    def either_or(
        self,
        either: Callable[[MultipleAttributesOperand], None],
        or_: Callable[[MultipleAttributesOperand], None],
    ) -> None:
        self._multiple_attributes_operand.either_or(
            lambda attributes: either(
                MultipleAttributesOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
            lambda attributes: or_(
                MultipleAttributesOperand._from_py_multiple_attributes_operand(
                    attributes
                )
            ),
        )

    def clone(self) -> MultipleAttributesOperand:
        return MultipleAttributesOperand._from_py_multiple_attributes_operand(
            self._multiple_attributes_operand.deep_clone()
        )

    @classmethod
    def _from_py_multiple_attributes_operand(
        cls, py_multiple_attributes_operand: PyMultipleAttributesOperand
    ) -> MultipleAttributesOperand:
        multiple_attributes_operand = cls()
        multiple_attributes_operand._multiple_attributes_operand = (
            py_multiple_attributes_operand
        )
        return multiple_attributes_operand


class SingleAttributeOperand:
    _single_attribute_operand: PySingleAttributeOperand

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
        either: Callable[[SingleAttributeOperand], None],
        or_: Callable[[SingleAttributeOperand], None],
    ) -> None:
        self._single_attribute_operand.either_or(
            lambda attribute: either(
                SingleAttributeOperand._from_py_single_attribute_operand(attribute)
            ),
            lambda attribute: or_(
                SingleAttributeOperand._from_py_single_attribute_operand(attribute)
            ),
        )

    def clone(self) -> SingleAttributeOperand:
        return SingleAttributeOperand._from_py_single_attribute_operand(
            self._single_attribute_operand.deep_clone()
        )

    @classmethod
    def _from_py_single_attribute_operand(
        cls, py_single_attribute_operand: PySingleAttributeOperand
    ) -> SingleAttributeOperand:
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

    def first(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.first()
        )

    def last(self) -> NodeIndexOperand:
        return NodeIndexOperand._from_py_node_index_operand(
            self._node_indices_operand.last()
        )

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


class NodeIndexOperand:
    _node_index_operand: PyNodeIndexOperand

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

    def first(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.first()
        )

    def last(self) -> EdgeIndexOperand:
        return EdgeIndexOperand._from_py_edge_index_operand(
            self._edge_indices_operand.last()
        )

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

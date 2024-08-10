from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from medmodels._medmodels import (
    PyEdgeOperand,
    PyEdgeValueOperand,
    PyEdgeValuesOperand,
    PyNodeOperand,
    PyNodeValueOperand,
    PyNodeValuesOperand,
)

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from medmodels.medrecord.types import (
        ComparisonOperand,
        Group,
        MedRecordAttribute,
        MedRecordValue,
    )

PyValueOperand: TypeAlias = Union[
    PyNodeValueOperand, PyEdgeValueOperand, MedRecordValue
]
PyValuesOperand: TypeAlias = Union[
    PyNodeValuesOperand, PyEdgeValuesOperand, List[MedRecordValue]
]
PyComparisonOperand: TypeAlias = Union[PyValueOperand, PyValuesOperand]


def _convert_comparison_operand(value: "ComparisonOperand") -> PyComparisonOperand:
    if isinstance(value, NodeValuesOperand):
        return value._node_values_operand
    elif isinstance(value, EdgeValuesOperand):
        return value._edge_values_operand
    elif isinstance(value, NodeValueOperand):
        return value._node_value_operand
    elif isinstance(value, EdgeValueOperand):
        return value._edge_value_operand
    else:
        return value


class NodeOperand:
    _node_operand: PyNodeOperand

    @classmethod
    def _from_py_node_operand(cls, node_operand: PyNodeOperand) -> "NodeOperand":
        instance = cls()
        instance._node_operand = node_operand
        return instance

    def in_group(self, group: Union[Group, List[Group]]):
        self._node_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ):
        self._node_operand.has_attribute(attribute)

    def outgoing_edges(self) -> "EdgeOperand":
        return EdgeOperand._from_py_edge_operand(self._node_operand.outgoing_edges())

    def incoming_edges(self) -> "EdgeOperand":
        return EdgeOperand._from_py_edge_operand(self._node_operand.incoming_edges())


class EdgeOperand:
    _edge_operand: PyEdgeOperand

    @classmethod
    def _from_py_edge_operand(cls, edge_operand: PyEdgeOperand) -> "EdgeOperand":
        instance = cls()
        instance._edge_operand = edge_operand
        return instance

    def attribute(self, attribute: MedRecordAttribute):
        return self._edge_operand.attribute(attribute)

    def in_group(self, group: Union[Group, List[Group]]):
        self._edge_operand.in_group(group)

    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ):
        self._edge_operand.has_attribute(attribute)

    def source_node(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._edge_operand.source_node())

    def target_node(self) -> NodeOperand:
        return NodeOperand._from_py_node_operand(self._edge_operand.target_node())


class NodeValuesOperand:
    _node_values_operand: PyNodeValuesOperand

    @classmethod
    def _from_py_node_values_operand(
        cls, node_values_operand: PyNodeValuesOperand
    ) -> "NodeValuesOperand":
        instance = cls()
        instance._node_values_operand = node_values_operand
        return instance


class EdgeValuesOperand:
    _edge_values_operand: PyEdgeValuesOperand

    @classmethod
    def _from_py_edge_values_operand(
        cls, edge_values_operand: PyEdgeValuesOperand
    ) -> "EdgeValuesOperand":
        instance = cls()
        instance._edge_values_operand = edge_values_operand
        return instance

    def max(self) -> "EdgeValueOperand":
        return EdgeValueOperand._from_py_edge_value_operand(
            self._edge_values_operand.max()
        )


class NodeValueOperand:
    _node_value_operand: PyNodeValueOperand

    @classmethod
    def _from_py_node_value_operand(
        cls, node_value_operand: PyNodeValueOperand
    ) -> "NodeValueOperand":
        instance = cls()
        instance._node_value_operand = node_value_operand
        return instance


class EdgeValueOperand:
    _edge_value_operand: PyEdgeValueOperand

    @classmethod
    def _from_py_edge_value_operand(
        cls, edge_value_operand: PyEdgeValueOperand
    ) -> "EdgeValueOperand":
        instance = cls()
        instance._edge_value_operand = edge_value_operand
        return instance

    def less_than(self, value: ComparisonOperand):
        self._edge_value_operand.less_than(_convert_comparison_operand(value))

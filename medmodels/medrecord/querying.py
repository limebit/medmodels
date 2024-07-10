from __future__ import annotations

from typing import List, Union

from medmodels._medmodels import (
    PyEdgeAttributeOperand,
    PyEdgeIndexOperand,
    PyEdgeOperand,
    PyEdgeOperation,
    PyNodeAttributeOperand,
    PyNodeIndexOperand,
    PyNodeOperand,
    PyNodeOperation,
    PyValueArithmeticOperation,
    PyValueTransformationOperation,
)
from medmodels.medrecord.types import (
    EdgeIndex,
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
)

ValueOperand = Union[
    MedRecordValue,
    MedRecordAttribute,
    PyValueArithmeticOperation,
    PyValueTransformationOperation,
]


class NodeOperation:
    _node_operation: PyNodeOperation

    def __init__(self, node_operation: PyNodeOperation):
        self._node_operation = node_operation

    def logical_and(self, operation: NodeOperation) -> NodeOperation:
        """
        Combines this NodeOperation with another using a logical AND, resulting in a
        new NodeOperation that is true only if both original operations are true.
        This method allows for the chaining of conditions to refine queries on nodes.

        Args:
            operation (NodeOperation): Another NodeOperation to be combined with the
                current one.

        Returns:
            NodeOperation: A new NodeOperation representing the logical AND of this
                operation with another.
        """
        return NodeOperation(
            self._node_operation.logical_and(operation._node_operation)
        )

    def __and__(self, operation: NodeOperation) -> NodeOperation:
        return self.logical_and(operation)

    def logical_or(self, operation: NodeOperation) -> NodeOperation:
        """
        Combines this NodeOperation with another using a logical OR, resulting in a
        new NodeOperation that is true if either of the original operations is true.
        This method enables the combination of conditions to expand queries on nodes.

        Args:
            operation (NodeOperation): Another NodeOperation to be combined with the
                current one.

        Returns:
            NodeOperation: A new NodeOperation representing the logical OR of this
                operation with another.
        """
        return NodeOperation(self._node_operation.logical_or(operation._node_operation))

    def __or__(self, operation: NodeOperation) -> NodeOperation:
        return self.logical_or(operation)

    def logical_xor(self, operation: NodeOperation) -> NodeOperation:
        """
        Combines this NodeOperation with another using a logical XOR, resulting in a
        new NodeOperation that is true only if exactly one of the original operations
        is true. This method is useful for creating conditions that must be
        exclusively true.

        Args:
            operation (NodeOperation): Another NodeOperation to be combined with the
                current one.

        Returns:
            NodeOperation: A new NodeOperation representing the logical XOR of this
                operation with another.
        """
        return NodeOperation(
            self._node_operation.logical_xor(operation._node_operation)
        )

    def __xor__(self, operation: NodeOperation) -> NodeOperation:
        return self.logical_xor(operation)

    def logical_not(self) -> NodeOperation:
        """
        Creates a new NodeOperation that is the logical NOT of this operation,
        inversing the current condition. This method is useful for negating a condition
        to create queries on nodes.

        Returns:
            NodeOperation: A new NodeOperation representing the logical NOT of
                this operation.
        """
        return NodeOperation(self._node_operation.logical_not())

    def __invert__(self) -> NodeOperation:
        return self.logical_not()


class EdgeOperation:
    _edge_operation: PyEdgeOperation

    def __init__(self, edge_operation: PyEdgeOperation) -> None:
        self._edge_operation = edge_operation

    def logical_and(self, operation: EdgeOperation) -> EdgeOperation:
        """
        Combines this EdgeOperation with another using a logical AND, resulting in a
        new EdgeOperation that is true only if both original operations are true.
        This method allows for the chaining of conditions to refine queries on nodes.

        Args:
            operation (EdgeOperation): Another EdgeOperation to be combined with the
                current one.

        Returns:
            EdgeOperation: A new EdgeOperation representing the logical AND of this
                operation with another.
        """
        return EdgeOperation(
            self._edge_operation.logical_and(operation._edge_operation)
        )

    def __and__(self, operation: EdgeOperation) -> EdgeOperation:
        return self.logical_and(operation)

    def logical_or(self, operation: EdgeOperation) -> EdgeOperation:
        """
        Combines this EdgeOperation with another using a logical OR, resulting in a
        new EdgeOperation that is true if either of the original operations is true.
        This method enables the combination of conditions to expand queries on nodes.

        Args:
            operation (EdgeOperation): Another EdgeOperation to be combined with the
                current one.

        Returns:
            EdgeOperation: A new EdgeOperation representing the logical OR of this
                operation with another.
        """
        return EdgeOperation(self._edge_operation.logical_or(operation._edge_operation))

    def __or__(self, operation: EdgeOperation) -> EdgeOperation:
        return self.logical_or(operation)

    def logical_xor(self, operation: EdgeOperation) -> EdgeOperation:
        """
        Combines this EdgeOperation with another using a logical XOR, resulting in a
        new EdgeOperation that is true only if exactly one of the original operations
        is true. This method is useful for creating conditions that must be
        exclusively true.

        Args:
            operation (EdgeOperation): Another EdgeOperation to be combined with the
                current one.

        Returns:
            EdgeOperation: A new EdgeOperation representing the logical XOR of this
                operation with another.
        """
        return EdgeOperation(
            self._edge_operation.logical_xor(operation._edge_operation)
        )

    def __xor__(self, operation: EdgeOperation) -> EdgeOperation:
        return self.logical_xor(operation)

    def logical_not(self) -> EdgeOperation:
        """
        Creates a new EdgeOperation that is the logical NOT of this operation,
        inversing the current condition. This method is useful for negating a condition
        to create queries on nodes.

        Returns:
            EdgeOperation: A new EdgeOperation representing the logical NOT of
                this operation.
        """
        return EdgeOperation(self._edge_operation.logical_not())

    def __invert__(self) -> EdgeOperation:
        return self.logical_not()


class NodeAttributeOperand:
    _node_attribute_operand: PyNodeAttributeOperand

    def __init__(self, node_attribute_operand: PyNodeAttributeOperand) -> None:
        self._node_attribute_operand = node_attribute_operand

    def greater(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is greater than the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the greater-than comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.greater(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.greater(operand))

    def __gt__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.greater(operand)

    def less(self, operand: Union[ValueOperand, NodeAttributeOperand]) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is less than the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the less-than comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.less(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.less(operand))

    def __lt__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.less(operand)

    def greater_or_equal(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is greater than or equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the
                greater-than-or-equal-to comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.greater_or_equal(
                    operand._node_attribute_operand
                )
            )

        return NodeOperation(self._node_attribute_operand.greater_or_equal(operand))

    def __ge__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.greater_or_equal(operand)

    def less_or_equal(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is less than or equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the
                less-than-or-equal-to comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.less_or_equal(
                    operand._node_attribute_operand
                )
            )

        return NodeOperation(self._node_attribute_operand.less_or_equal(operand))

    def __le__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.less_or_equal(operand)

    def equal(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.y

        Returns:
            NodeOperation: A NodeOperation representing the equality comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.equal(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.equal(operand))

    def __eq__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.equal(operand)

    def not_equal(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is not equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the not-equal comparison.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.not_equal(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.not_equal(operand))

    def __ne__(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        return self.not_equal(operand)

    def is_in(self, values: List[MedRecordValue]) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is found within the specified list of values.

        Args:
            values (List[MedRecordValue]): The list of values to check the
                attribute against.

        Returns:
            NodeOperation: A NodeOperation representing the is-in comparison.
        """
        return NodeOperation(self._node_attribute_operand.is_in(values))

    def not_in(self, values: List[MedRecordValue]) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand is not found within the specified list of values.

        Args:
            values (List[MedRecordValue]): The list of values to check the
                attribute against.

        Returns:
            NodeOperation: A NodeOperation representing the not-in comparison.
        """
        return NodeOperation(self._node_attribute_operand.not_in(values))

    def starts_with(self, operand: ValueOperand) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand starts with the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare
                the starting sequence against.

        Returns:
            NodeOperation: A NodeOperation representing the starts-with condition.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.starts_with(
                    operand._node_attribute_operand
                )
            )

        return NodeOperation(self._node_attribute_operand.starts_with(operand))

    def ends_with(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand ends with the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare
                the ending sequence against.

        Returns:
            NodeOperation: A NodeOperation representing the ends-with condition.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.ends_with(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.ends_with(operand))

    def contains(
        self, operand: Union[ValueOperand, NodeAttributeOperand]
    ) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the attribute represented
        by this operand contains the specified value or operand within it.

        Args:
            operand (ValueOperand): The value or operand to check for containment.

        Returns:
            NodeOperation: A NodeOperation representing the contains condition.
        """
        if isinstance(operand, NodeAttributeOperand):
            return NodeOperation(
                self._node_attribute_operand.contains(operand._node_attribute_operand)
            )

        return NodeOperation(self._node_attribute_operand.contains(operand))

    def add(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the sum of the attribute's value
        and the specified value.

        Args:
            value (MedRecordValue): The value to add to the attribute's value.

        Returns:
            ValueOperand: The result of the addition operation.
        """
        return self._node_attribute_operand.add(value)

    def __add__(self, value: MedRecordValue) -> ValueOperand:
        return self.add(value)

    def sub(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the difference between the attribute's
        value and the specified value.

        Args:
            value (MedRecordValue): The value to subtract from the attribute's value.

        Returns:
            ValueOperand: The result of the subtraction operation.
        """
        return self._node_attribute_operand.sub(value)

    def __sub__(self, value: MedRecordValue) -> ValueOperand:
        return self.sub(value)

    def mul(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the product of the attribute's value
        and the specified value.

        Args:
            value (MedRecordValue): The value to multiply the attribute's value by.

        Returns:
            ValueOperand: The result of the multiplication operation.
        """
        return self._node_attribute_operand.mul(value)

    def __mul__(self, value: MedRecordValue) -> ValueOperand:
        return self.mul(value)

    def div(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the division of the attribute's value
        by the specified value.

        Args:
            value (MedRecordValue): The value to divide the attribute's value by.

        Returns:
            ValueOperand: The result of the division operation.
        """
        return self._node_attribute_operand.div(value)

    def __truediv__(self, value: MedRecordValue) -> ValueOperand:
        return self.div(value)

    def pow(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of raising the attribute's
        value to the power of the specified value.

        Args:
            value (MedRecordValue): The value to raise the attribute's value to.

        Returns:
            ValueOperand: The result of the exponentiation operation.
        """
        return self._node_attribute_operand.pow(value)

    def __pow__(self, value: MedRecordValue) -> ValueOperand:
        return self.pow(value)

    def mod(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the remainder of dividing the
        attribute's value by the specified value.

        Args:
        value (MedRecordValue): The value to divide the attribute's value by.

        Returns:
        ValueOperand: The result of the modulo operation.
        """
        return self._node_attribute_operand.mod(value)

    def __mod__(self, value: MedRecordValue) -> ValueOperand:
        return self.mod(value)

    def round(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of rounding the
        attribute's value.

        Returns:
            ValueOperand: The result of the rounding operation.
        """
        return self._node_attribute_operand.round()

    def ceil(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of applying the ceiling
        function to the attribute's value, effectively rounding it up to the nearest
        whole number.

        Returns:
            ValueOperand: The result of the ceiling operation.
        """
        return self._node_attribute_operand.ceil()

    def floor(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of applying the floor
        function to the attribute's value, effectively rounding it down to the nearest
        whole number.

        Returns:
            ValueOperand: The result of the floor operation.
        """
        return self._node_attribute_operand.floor()

    def abs(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the absolute value of the
        attribute's value.

        Returns:
            ValueOperand: The absolute value of the attribute's value.
        """
        return self._node_attribute_operand.abs()

    def sqrt(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the square root of the
        attribute's value.

        Returns:
            ValueOperand: The square root of the attribute's value.
        """
        return self._node_attribute_operand.sqrt()

    def trim(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        both ends of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with leading and trailing
                whitespace removed.
        """
        return self._node_attribute_operand.trim()

    def trim_start(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        the start (left side) of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with leading whitespace removed.
        """
        return self._node_attribute_operand.trim_start()

    def trim_end(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        the end (right side) of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with trailing whitespace removed.
        """
        return self._node_attribute_operand.trim_end()

    def lowercase(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of converting all characters
        in the attribute's value to lowercase.

        Returns:
            ValueOperand: The attribute's value in lowercase letters.
        """
        return self._node_attribute_operand.lowercase()

    def uppercase(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of converting all characters
        in the attribute's value to uppercase.

        Returns:
            ValueOperand: The attribute's value in uppercase letters.
        """
        return self._node_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of slicing the attribute's
        value using the specified start and end indices.

        Args:
            start (int): The index at which to start the slice.
            end (int): The index at which to end the slice.

        Returns:
            ValueOperand: The attribute's value with the specified slice applied.
        """
        return self._node_attribute_operand.slice(start, end)


class EdgeAttributeOperand:
    _edge_attribute_operand: PyEdgeAttributeOperand

    def __init__(self, edge_attribute_operand: PyEdgeAttributeOperand) -> None:
        self._edge_attribute_operand = edge_attribute_operand

    def greater(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is greater than the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the greater-than comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.greater(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.greater(operand))

    def __gt__(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        return self.greater(operand)

    def less(self, operand: Union[ValueOperand, EdgeAttributeOperand]) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is less than the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the less-than comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.less(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.less(operand))

    def __lt__(self, operand: ValueOperand) -> EdgeOperation:
        return self.less(operand)

    def greater_or_equal(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is greater than or equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the
                greater-than-or-equal-to comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.greater_or_equal(
                    operand._edge_attribute_operand
                )
            )

        return EdgeOperation(self._edge_attribute_operand.greater_or_equal(operand))

    def __ge__(self, operand: ValueOperand) -> EdgeOperation:
        return self.greater_or_equal(operand)

    def less_or_equal(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is less than or equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the
                less-than-or-equal-to comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.less_or_equal(
                    operand._edge_attribute_operand
                )
            )

        return EdgeOperation(self._edge_attribute_operand.less_or_equal(operand))

    def __le__(self, operand: ValueOperand) -> EdgeOperation:
        return self.less_or_equal(operand)

    def equal(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the equality comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.equal(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.equal(operand))

    def __eq__(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        return self.equal(operand)

    def not_equal(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is not equal to the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the not-equal comparison.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.not_equal(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.not_equal(operand))

    def __ne__(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        return self.not_equal(operand)

    def is_in(self, values: List[MedRecordValue]) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is found within the specified list of values.

        Args:
            values (List[MedRecordValue]): The list of values to check the
                attribute against.

        Returns:
            EdgeOperation: A EdgeOperation representing the is-in comparison.
        """
        return EdgeOperation(self._edge_attribute_operand.is_in(values))

    def not_in(self, values: List[MedRecordValue]) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand is not found within the specified list of values.

        Args:
            values (List[MedRecordValue]): The list of values to check the
                attribute against.

        Returns:
            EdgeOperation: A EdgeOperation representing the not-in comparison.
        """
        return EdgeOperation(self._edge_attribute_operand.not_in(values))

    def starts_with(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand starts with the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare
                the starting sequence against.

        Returns:
            EdgeOperation: A EdgeOperation representing the starts-with condition.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.starts_with(
                    operand._edge_attribute_operand
                )
            )

        return EdgeOperation(self._edge_attribute_operand.starts_with(operand))

    def ends_with(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand ends with the specified value or operand.

        Args:
            operand (ValueOperand): The value or operand to compare
                the ending sequence against.

        Returns:
            EdgeOperation: A EdgeOperation representing the ends-with condition.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.ends_with(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.ends_with(operand))

    def contains(
        self, operand: Union[ValueOperand, EdgeAttributeOperand]
    ) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the attribute represented
        by this operand contains the specified value or operand within it.

        Args:
            operand (ValueOperand): The value or operand to check for containment.

        Returns:
            EdgeOperation: A EdgeOperation representing the contains condition.
        """
        if isinstance(operand, EdgeAttributeOperand):
            return EdgeOperation(
                self._edge_attribute_operand.contains(operand._edge_attribute_operand)
            )

        return EdgeOperation(self._edge_attribute_operand.contains(operand))

    def add(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the sum of the attribute's value
        and the specified value.

        Args:
            value (MedRecordValue): The value to add to the attribute's value.

        Returns:
            ValueOperand: The result of the addition operation.
        """
        return self._edge_attribute_operand.add(value)

    def __add__(self, value: MedRecordValue) -> ValueOperand:
        return self.add(value)

    def sub(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the difference between the attribute's
        value and the specified value.

        Args:
            value (MedRecordValue): The value to subtract from the attribute's value.

        Returns:
            ValueOperand: The result of the subtraction operation.
        """
        return self._edge_attribute_operand.sub(value)

    def __sub__(self, value: MedRecordValue) -> ValueOperand:
        return self.sub(value)

    def mul(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the product of the attribute's value
        and the specified value.

        Args:
            value (MedRecordValue): The value to multiply the attribute's value by.

        Returns:
            ValueOperand: The result of the multiplication operation.
        """
        return self._edge_attribute_operand.mul(value)

    def __mul__(self, value: MedRecordValue) -> ValueOperand:
        return self.mul(value)

    def div(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the division of the attribute's value
        by the specified value.

        Args:
            value (MedRecordValue): The value to divide the attribute's value by.

        Returns:
            ValueOperand: The result of the division operation.
        """
        return self._edge_attribute_operand.div(value)

    def __truediv__(self, value: MedRecordValue) -> ValueOperand:
        return self.div(value)

    def pow(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of raising the attribute's
        value to the power of the specified value.

        Args:
            value (MedRecordValue): The value to raise the attribute's value to.

        Returns:
            ValueOperand: The result of the exponentiation operation.
        """
        return self._edge_attribute_operand.pow(value)

    def __pow__(self, value: MedRecordValue) -> ValueOperand:
        return self.pow(value)

    def mod(self, value: MedRecordValue) -> ValueOperand:
        """
        Creates a new ValueOperand representing the remainder of dividing the
        attribute's value by the specified value.

        Args:
            value (MedRecordValue): The value to divide the attribute's value by.

        Returns:
            ValueOperand: The result of the modulo operation.
        """
        return self._edge_attribute_operand.mod(value)

    def __mod__(self, value: MedRecordValue) -> ValueOperand:
        return self.mod(value)

    def round(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of rounding the
        attribute's value.

        Returns:
            ValueOperand: The result of the rounding operation.
        """
        return self._edge_attribute_operand.round()

    def ceil(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of applying the ceiling
        function to the attribute's value, effectively rounding it up to the nearest
        whole number.

        Returns:
            ValueOperand: The result of the ceiling operation.
        """
        return self._edge_attribute_operand.ceil()

    def floor(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of applying the floor
        function to the attribute's value, effectively rounding it down to the nearest
        whole number.

        Returns:
            ValueOperand: The result of the floor operation.
        """
        return self._edge_attribute_operand.floor()

    def abs(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the absolute value of the
        attribute's value.

        Returns:
            ValueOperand: The absolute value of the attribute's value.
        """
        return self._edge_attribute_operand.abs()

    def sqrt(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the square root of the
        attribute's value.

        Returns:
            ValueOperand: The square root of the attribute's value.
        """
        return self._edge_attribute_operand.sqrt()

    def trim(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        both ends of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with leading and trailing
                whitespace removed.
        """
        return self._edge_attribute_operand.trim()

    def trim_start(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        the start (left side) of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with leading whitespace removed.
        """
        return self._edge_attribute_operand.trim_start()

    def trim_end(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of trimming whitespace from
        the end (right side) of the attribute's value.

        Returns:
            ValueOperand: The attribute's value with trailing whitespace removed.
        """
        return self._edge_attribute_operand.trim_end()

    def lowercase(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of converting all characters
        in the attribute's value to lowercase.

        Returns:
            ValueOperand: The attribute's value in lowercase letters.
        """
        return self._edge_attribute_operand.lowercase()

    def uppercase(self) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of converting all characters
        in the attribute's value to uppercase.

        Returns:
            ValueOperand: The attribute's value in uppercase letters.
        """
        return self._edge_attribute_operand.uppercase()

    def slice(self, start: int, end: int) -> ValueOperand:
        """
        Creates a new ValueOperand representing the result of slicing the attribute's
        value using the specified start and end indices.

        Args:
            start (int): The index at which to start the slice.
            end (int): The index at which to end the slice.

        Returns:
            ValueOperand: The attribute's value with the specified slice applied.
        """
        return self._edge_attribute_operand.slice(start, end)


class NodeIndexOperand:
    _node_index_operand: PyNodeIndexOperand

    def __init__(self, node_index_operand: PyNodeIndexOperand) -> None:
        self._node_index_operand = node_index_operand

    def greater(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is greater
        than the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the greater-than comparison.
        """
        return NodeOperation(self._node_index_operand.greater(operand))

    def __gt__(self, operand: NodeIndex) -> NodeOperation:
        return self.greater(operand)

    def less(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is less
        than the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the less-than comparison.
        """
        return NodeOperation(self._node_index_operand.less(operand))

    def __lt__(self, operand: NodeIndex) -> NodeOperation:
        return self.less(operand)

    def greater_or_equal(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is
        greater than or equal to the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the
                greater-than-or-equal-to comparison.
        """
        return NodeOperation(self._node_index_operand.greater_or_equal(operand))

    def __ge__(self, operand: NodeIndex) -> NodeOperation:
        return self.greater_or_equal(operand)

    def less_or_equal(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is
        less than or equal to the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the
                less-than-or-equal-to comparison.
        """
        return NodeOperation(self._node_index_operand.less_or_equal(operand))

    def __le__(self, operand: NodeIndex) -> NodeOperation:
        return self.less_or_equal(operand)

    def equal(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is equal to
        the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the equality comparison.
        """
        return NodeOperation(self._node_index_operand.equal(operand))

    def __eq__(self, operand: NodeIndex) -> NodeOperation:
        return self.equal(operand)

    def not_equal(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is not equal to
        the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the not-equal comparison.
        """
        return NodeOperation(self._node_index_operand.not_equal(operand))

    def __ne__(self, operand: NodeIndex) -> NodeOperation:
        return self.not_equal(operand)

    def is_in(self, values: List[NodeIndex]) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is found within
        the list of indices.

        Args:
            values (List[NodeIndex]): The list of indices to check the node index
                against.

        Returns:
            NodeOperation: A NodeOperation representing the is-in comparison.
        """
        return NodeOperation(self._node_index_operand.is_in(values))

    def not_in(self, values: List[NodeIndex]) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index is
        not found within the list of indices.

        Args:
            values (List[NodeIndex]): The list of indices to check the node index
                against.

        Returns:
            NodeOperation: A NodeOperation representing the not-in comparison.
        """
        return NodeOperation(self._node_index_operand.not_in(values))

    def starts_with(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index starts with
        the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the starts-with condition.
        """
        return NodeOperation(self._node_index_operand.starts_with(operand))

    def ends_with(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index ends with
        the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the ends-with condition.
        """
        return NodeOperation(self._node_index_operand.ends_with(operand))

    def contains(self, operand: NodeIndex) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node index contains
        the specified index.

        Args:
            operand (NodeIndex): The index to compare against.

        Returns:
            NodeOperation: A NodeOperation representing the contains condition.
        """
        return NodeOperation(self._node_index_operand.contains(operand))


class EdgeIndexOperand:
    _edge_index_operand: PyEdgeIndexOperand

    def __init__(self, edge_index_operand: PyEdgeIndexOperand) -> None:
        self._edge_index_operand = edge_index_operand

    def greater(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is greater
        than the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the greater-than comparison.
        """
        return EdgeOperation(self._edge_index_operand.greater(operand))

    def __gt__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.greater(operand)

    def less(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is less
        than the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the less-than comparison.
        """
        return EdgeOperation(self._edge_index_operand.less(operand))

    def __lt__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.less(operand)

    def greater_or_equal(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is
        greater than or equal to the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the
                greater-than-or-equal-to comparison.
        """
        return EdgeOperation(self._edge_index_operand.greater_or_equal(operand))

    def __ge__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.greater_or_equal(operand)

    def less_or_equal(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is
        less than or equal to the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the
                less-than-or-equal-to comparison.
        """
        return EdgeOperation(self._edge_index_operand.less_or_equal(operand))

    def __le__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.less_or_equal(operand)

    def equal(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is equal to
        the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the equality comparison.
        """
        return EdgeOperation(self._edge_index_operand.equal(operand))

    def __eq__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.equal(operand)

    def not_equal(self, operand: EdgeIndex) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is not equal to
        the specified index.

        Args:
            operand (EdgeIndex): The index to compare against.

        Returns:
            EdgeOperation: A EdgeOperation representing the not-equal comparison.
        """
        return EdgeOperation(self._edge_index_operand.not_equal(operand))

    def __ne__(self, operand: EdgeIndex) -> EdgeOperation:
        return self.not_equal(operand)

    def is_in(self, values: List[EdgeIndex]) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is found within
        the list of indices.

        Args:
            values (List[EdgeIndex]): The list of indices to check the edge index
                against.

        Returns:
            EdgeOperation: A EdgeOperation representing the is-in comparison.
        """
        return EdgeOperation(self._edge_index_operand.is_in(values))

    def not_in(self, values: List[EdgeIndex]) -> EdgeOperation:
        """
        Creates a EdgeOperation that evaluates to true if the edge index is
        not found within the list of indices.

        Args:
            values (List[EdgeIndex]): The list of indices to check the edge index
                against.

        Returns:
            EdgeOperation: A EdgeOperation representing the not-in comparison.
        """
        return EdgeOperation(self._edge_index_operand.not_in(values))


class NodeOperand:
    _node_operand: PyNodeOperand

    def __init__(self) -> None:
        self._node_operand = PyNodeOperand()

    def in_group(self, operand: Group) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node is part of the
        specified group.

        Args:
            operand (Group): The group to check the node against.

        Returns:
            NodeOperation: A NodeOperation indicating if the node is part of the
                specified group.
        """
        return NodeOperation(self._node_operand.in_group(operand))

    def has_attribute(self, operand: MedRecordAttribute) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node has the
        specified attribute.

        Args:
            operand (MedRecordAttribute): The attribute to check on the node.

        Returns:
            NodeOperation: A NodeOperation indicating if the node has the
                specified attribute.
        """
        return NodeOperation(self._node_operand.has_attribute(operand))

    def has_outgoing_edge_with(self, operation: EdgeOperation) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node has an
        outgoing edge that satisfies the specified EdgeOperation.

        Args:
            operation (EdgeOperation): An EdgeOperation to evaluate against
                outgoing edges.

        Returns:
            NodeOperation: A NodeOperation indicating if the node has an
                outgoing edge satisfying the specified operation.
        """
        return NodeOperation(
            self._node_operand.has_outgoing_edge_with(operation._edge_operation)
        )

    def has_incoming_edge_with(self, operation: EdgeOperation) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node has an
        incoming edge that satisfies the specified EdgeOperation.

        Args:
            operation (EdgeOperation): An EdgeOperation to evaluate against
                incoming edges.

        Returns:
            NodeOperation: A NodeOperation indicating if the node has an
                incoming edge satisfying the specified operation.
        """
        return NodeOperation(
            self._node_operand.has_incoming_edge_with(operation._edge_operation)
        )

    def has_edge_with(self, operation: EdgeOperation) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node has any
        edge (incoming or outgoing) that satisfies the specified EdgeOperation.

        Args:
            operation (EdgeOperation): An EdgeOperation to evaluate against
                edges connected to the node.

        Returns:
            NodeOperation: A NodeOperation indicating if the node has any edge
                satisfying the specified operation.
        """
        return NodeOperation(
            self._node_operand.has_edge_with(operation._edge_operation)
        )

    def has_neighbor_with(self, operation: NodeOperation) -> NodeOperation:
        """
        Creates a NodeOperation that evaluates to true if the node has a
        neighboring node that satisfies the specified NodeOperation.

        Args:
            operation (NodeOperation): A NodeOperation to evaluate against
                neighboring nodes.

        Returns:
            NodeOperation: A NodeOperation indicating if the node has a neighboring node
                satisfying the specified operation.
        """
        return NodeOperation(
            self._node_operand.has_neighbor_with(operation._node_operation)
        )

    def attribute(self, attribute: MedRecordAttribute) -> NodeAttributeOperand:
        """
        Accesses an NodeAttributeOperand for the specified attribute,
        allowing for the creation of operations based on node attributes.

        Args:
            attribute (MedRecordAttribute): The attribute of the node to perform
                operations on.

        Returns:
            NodeAttributeOperand: An operand that represents the specified node
                attribute, enabling further operations such as comparisons and
                arithmetic operations.
        """
        return NodeAttributeOperand(self._node_operand.attribute(attribute))

    def index(self) -> NodeIndexOperand:
        """
        Accesses an NodeIndexOperand, allowing for the creation of operations based on
        the node index.

        Returns:
            NodeIndexOperand: An operand that represents the specified node
                index, enabling further operations such as comparisons and
                arithmetic operations.
        """
        return NodeIndexOperand(self._node_operand.index())


def node() -> NodeOperand:
    """
    Factory function to create and return a new NodeOperand instance.

    Returns:
        NodeOperand: An instance of NodeOperand for constructing node-based operations.
    """
    return NodeOperand()


class EdgeOperand:
    _edge_operand: PyEdgeOperand

    def __init__(self) -> None:
        self._edge_operand = PyEdgeOperand()

    def connected_target(self, operand: NodeIndex) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge is connected to a
        target node with the specified index.

        Args:
            operand (NodeIndex): The index of the target node to check for a connection.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the edge is connected to the
                specified target node.
        """
        return EdgeOperation(self._edge_operand.connected_target(operand))

    def connected_source(self, operand: NodeIndex) -> EdgeOperation:
        """
        Generates an EdgeOperation that evaluates to true if the edge originates from a
        source node with the given index.

        Args:
            operand (NodeIndex): The index of the source node to check for a connection.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the edge is connected from the
                specified source node.
        """
        return EdgeOperation(self._edge_operand.connected_source(operand))

    def connected(self, operand: NodeIndex) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge is connected
        to or from a node with the specified index.

        Args:
            operand (NodeIndex): The index of the node to check for a connection.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the edge is connected to the
                specified node.
        """
        return EdgeOperation(self._edge_operand.connected(operand))

    def in_group(self, operand: Group) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge is part of the
        specified group.

        Args:
            operand (Group): The group to check the edge against.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the edge is part of the
                specified group.
        """
        return EdgeOperation(self._edge_operand.in_group(operand))

    def has_attribute(self, operand: MedRecordAttribute) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge has the
        specified attribute.

        Args:
            operand (MedRecordAttribute): The attribute to check on the edge.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the edge has the
                specified attribute.
        """
        return EdgeOperation(self._edge_operand.has_attribute(operand))

    def connected_source_with(self, operation: NodeOperation) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge originates from a
        source node that satisfies the specified NodeOperation.

        Args:
            operation (NodeOperation): A NodeOperation to evaluate against the
                source node.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the source node of the
                edge satisfies the specified operation.
        """
        return EdgeOperation(
            self._edge_operand.connected_source_with(operation._node_operation)
        )

    def connected_target_with(self, operation: NodeOperation) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge is connected to a
        target node that satisfies the specified NodeOperation.

        Args:
            operation (NodeOperation): A NodeOperation to evaluate against the
            target node.

        Returns:
            EdgeOperation: An EdgeOperation indicating if the target node of the
                edge satisfies the specified operation.
        """
        return EdgeOperation(
            self._edge_operand.connected_target_with(operation._node_operation)
        )

    def connected_with(self, operation: NodeOperation) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if the edge is connected
        to or from a node that satisfies the specified NodeOperation.

        Args:
            operation (NodeOperation): A NodeOperation to evaluate against the
                connected node.

        Returns:
            EdgeOperation: An EdgeOperation indicating if either the source or
                target node of the edge satisfies the specified operation.
        """
        return EdgeOperation(
            self._edge_operand.connected_with(operation._node_operation)
        )

    def has_parallel_edges_with(self, operation: EdgeOperation) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if there are parallel edges that
        satisfy the specified EdgeOperation.

        Args:
            operation (EdgeOperation): An EdgeOperation to evaluate against
                parallel edges.

        Returns:
            EdgeOperation: An EdgeOperation indicating if there are parallel edges
                satisfying the specified operation.
        """
        return EdgeOperation(
            self._edge_operand.has_parallel_edges_with(operation._edge_operation)
        )

    def has_parallel_edges_with_self_comparison(
        self, operation: EdgeOperation
    ) -> EdgeOperation:
        """
        Creates an EdgeOperation that evaluates to true if there are parallel edges that
        satisfy the specified EdgeOperation.

        Using `edge().attribute(...)` in the operation will compare to the attribute of
        this edge, not the parallel edge.

        Args:
            operation (EdgeOperation): An EdgeOperation to evaluate against
                parallel edges.

        Returns:
            EdgeOperation: An EdgeOperation indicating if there are parallel edges
                satisfying the specified operation.
        """
        return EdgeOperation(
            self._edge_operand.has_parallel_edges_with_self_comparison(
                operation._edge_operation
            )
        )

    def attribute(self, attribute: MedRecordAttribute) -> EdgeAttributeOperand:
        """
        Accesses an EdgeAttributeOperand for the specified attribute,
        allowing for the creation of operations based on edge attributes.

        Args:
            attribute (MedRecordAttribute): The attribute of the edge to perform
                operations on.

        Returns:
            EdgeAttributeOperand: An operand that represents the specified edge
                attribute, enabling further operations such as comparisons and
                arithmetic operations.
        """
        return EdgeAttributeOperand(self._edge_operand.attribute(attribute))

    def index(self) -> EdgeIndexOperand:
        """
        Accesses an EdgeIndexOperand, allowing for the creation of operations based on
        the edge index.

        Returns:
            EdgeIndexOperand: An operand that represents the specified edge
                index, enabling further operations such as comparisons and
                arithmetic operations.
        """
        return EdgeIndexOperand(self._edge_operand.index())


def edge() -> EdgeOperand:
    """
    Factory function to create and return a new EdgeOperand instance.

    Returns:
        EdgeOperand: An instance of EdgeOperand for constructing edge-based operations.
    """
    return EdgeOperand()

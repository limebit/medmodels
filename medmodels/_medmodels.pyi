from typing import Dict, List, Optional, Union

from medmodels.medrecord.types import (
    Attributes,
    EdgeIndex,
    EdgeIndexInputList,
    Group,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
    NodeIndexInputList,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
)

ValueOperand = Union[
    MedRecordValue,
    MedRecordAttribute,
    PyValueArithmeticOperation,
    PyValueTransformationOperation,
]

class PyMedRecord:
    nodes: List[NodeIndex]
    edges: List[EdgeIndex]
    groups: List[Group]

    def __init__(self) -> None: ...
    @staticmethod
    def from_tuples(
        nodes: List[tuple[NodeIndex, Attributes]],
        edges: Optional[List[tuple[NodeIndex, NodeIndex, Attributes]]],
    ) -> PyMedRecord: ...
    @staticmethod
    def from_dataframes(
        nodes_dataframes: List[PolarsNodeDataFrameInput],
        edges_dataframes: List[PolarsEdgeDataFrameInput],
    ) -> PyMedRecord: ...
    @staticmethod
    def from_nodes_dataframes(
        nodes_dataframes: List[PolarsNodeDataFrameInput],
    ) -> PyMedRecord: ...
    @staticmethod
    def from_ron(path: str) -> PyMedRecord: ...
    def to_ron(self, path: str) -> None: ...
    def node(self, node_index: NodeIndexInputList) -> Dict[NodeIndex, Attributes]: ...
    def edge(self, edge_index: EdgeIndexInputList) -> Dict[EdgeIndex, Attributes]: ...
    def group(self, group: List[Group]) -> Dict[Group, List[NodeIndex]]: ...
    def outgoing_edges(
        self, node_index: List[NodeIndex]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...
    def incoming_edges(
        self, node_index: List[NodeIndex]
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...
    def edge_endpoints(
        self, edge_index: List[EdgeIndex]
    ) -> Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]: ...
    def edges_connecting(
        self, source_node_index: List[NodeIndex], target_node_index: List[NodeIndex]
    ) -> List[EdgeIndex]: ...
    def add_node(self, node_index: NodeIndex, attributes: Attributes) -> None: ...
    def remove_node(
        self, node_index: List[NodeIndex]
    ) -> Dict[NodeIndex, Attributes]: ...
    def replace_node_attributes(
        self, attributes: Attributes, node_index: NodeIndexInputList
    ) -> None: ...
    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        value: MedRecordValue,
        node_index: NodeIndexInputList,
    ) -> None: ...
    def remove_node_attribute(
        self, attribute: MedRecordAttribute, node_index: NodeIndexInputList
    ) -> None: ...
    def add_nodes(self, nodes: List[tuple[NodeIndex, Attributes]]) -> None: ...
    def add_nodes_dataframes(
        self, nodes_dataframe: List[PolarsNodeDataFrameInput]
    ) -> None: ...
    def add_edge(
        self,
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
        attributes: Attributes,
    ) -> EdgeIndex: ...
    def remove_edge(
        self, edge_index: List[EdgeIndex]
    ) -> Dict[EdgeIndex, Attributes]: ...
    def replace_edge_attributes(
        self, attributes: Attributes, edge_index: List[EdgeIndex]
    ) -> None: ...
    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        value: MedRecordValue,
        edge_index: List[EdgeIndex],
    ) -> None: ...
    def remove_edge_attribute(
        self, attribute: MedRecordAttribute, edge_index: List[EdgeIndex]
    ) -> None: ...
    def add_edges(
        self, edges: List[tuple[NodeIndex, NodeIndex, Attributes]]
    ) -> List[EdgeIndex]: ...
    def add_edges_dataframes(
        self, edges_dataframe: List[PolarsEdgeDataFrameInput]
    ) -> List[EdgeIndex]: ...
    def add_group(
        self, group: Group, node_indices_to_add: Optional[List[NodeIndex]]
    ) -> None: ...
    def remove_group(self, group: List[Group]) -> None: ...
    def add_node_to_group(self, group: Group, node_index: List[NodeIndex]) -> None: ...
    def remove_node_from_group(
        self, group: Group, node_index: List[NodeIndex]
    ) -> None: ...
    def groups_of_node(
        self, node_index: List[NodeIndex]
    ) -> Dict[NodeIndex, List[Group]]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def group_count(self) -> int: ...
    def contains_node(self, node_index: NodeIndex) -> bool: ...
    def contains_edge(self, edge_index: EdgeIndex) -> bool: ...
    def contains_group(self, group: Group) -> bool: ...
    def neighbors(
        self, node_index: List[NodeIndex]
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...
    def clear(self) -> None: ...
    def select_nodes(self, operation: PyNodeOperation) -> List[NodeIndex]: ...
    def select_edges(self, operation: PyEdgeOperation) -> List[EdgeIndex]: ...

class PyValueArithmeticOperation: ...
class PyValueTransformationOperation: ...

class PyNodeOperation:
    def logical_and(self, operation: PyNodeOperation) -> PyNodeOperation: ...
    def logical_or(self, operation: PyNodeOperation) -> PyNodeOperation: ...
    def logical_xor(self, operation: PyNodeOperation) -> PyNodeOperation: ...
    def logical_not(self) -> PyNodeOperation: ...

class PyEdgeOperation:
    def logical_and(self, operation: PyEdgeOperation) -> PyEdgeOperation: ...
    def logical_or(self, operation: PyEdgeOperation) -> PyEdgeOperation: ...
    def logical_xor(self, operation: PyEdgeOperation) -> PyEdgeOperation: ...
    def logical_not(self) -> PyEdgeOperation: ...

class PyNodeAttributeOperand:
    def greater(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def less(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def greater_or_equal(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def less_or_equal(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def equal(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def not_equal(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def is_in(self, operands: List[MedRecordValue]) -> PyNodeOperation: ...
    def not_in(self, operands: List[MedRecordValue]) -> PyNodeOperation: ...
    def starts_with(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def ends_with(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def contains(
        self, operand: Union[ValueOperand, PyNodeAttributeOperand]
    ) -> PyNodeOperation: ...
    def add(self, value: MedRecordValue) -> ValueOperand: ...
    def sub(self, value: MedRecordValue) -> ValueOperand: ...
    def mul(self, value: MedRecordValue) -> ValueOperand: ...
    def div(self, value: MedRecordValue) -> ValueOperand: ...
    def pow(self, value: MedRecordValue) -> ValueOperand: ...
    def mod(self, value: MedRecordValue) -> ValueOperand: ...
    def round(self) -> ValueOperand: ...
    def ceil(self) -> ValueOperand: ...
    def floor(self) -> ValueOperand: ...
    def abs(self) -> ValueOperand: ...
    def sqrt(self) -> ValueOperand: ...
    def trim(self) -> ValueOperand: ...
    def trim_start(self) -> ValueOperand: ...
    def trim_end(self) -> ValueOperand: ...
    def lowercase(self) -> ValueOperand: ...
    def uppercase(self) -> ValueOperand: ...
    def slice(self, start: int, end: int) -> ValueOperand: ...

class PyEdgeAttributeOperand:
    def greater(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def less(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def greater_or_equal(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def less_or_equal(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def equal(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def not_equal(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def is_in(self, operands: List[MedRecordValue]) -> PyEdgeOperation: ...
    def not_in(self, operands: List[MedRecordValue]) -> PyEdgeOperation: ...
    def starts_with(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def ends_with(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def contains(
        self, operand: Union[ValueOperand, PyEdgeAttributeOperand]
    ) -> PyEdgeOperation: ...
    def add(self, value: MedRecordValue) -> ValueOperand: ...
    def sub(self, value: MedRecordValue) -> ValueOperand: ...
    def mul(self, value: MedRecordValue) -> ValueOperand: ...
    def div(self, value: MedRecordValue) -> ValueOperand: ...
    def pow(self, value: MedRecordValue) -> ValueOperand: ...
    def mod(self, value: MedRecordValue) -> ValueOperand: ...
    def round(self) -> ValueOperand: ...
    def ceil(self) -> ValueOperand: ...
    def floor(self) -> ValueOperand: ...
    def abs(self) -> ValueOperand: ...
    def sqrt(self) -> ValueOperand: ...
    def trim(self) -> ValueOperand: ...
    def trim_start(self) -> ValueOperand: ...
    def trim_end(self) -> ValueOperand: ...
    def lowercase(self) -> ValueOperand: ...
    def uppercase(self) -> ValueOperand: ...
    def slice(self, start: int, end: int) -> ValueOperand: ...

class PyNodeIndexOperand:
    def greater(self, operand: NodeIndex) -> PyNodeOperation: ...
    def less(self, operand: NodeIndex) -> PyNodeOperation: ...
    def greater_or_equal(self, operand: NodeIndex) -> PyNodeOperation: ...
    def less_or_equal(self, operand: NodeIndex) -> PyNodeOperation: ...
    def equal(self, operand: NodeIndex) -> PyNodeOperation: ...
    def not_equal(self, operand: NodeIndex) -> PyNodeOperation: ...
    def is_in(self, operand: List[NodeIndex]) -> PyNodeOperation: ...
    def not_in(self, operand: List[NodeIndex]) -> PyNodeOperation: ...
    def starts_with(self, operand: NodeIndex) -> PyNodeOperation: ...
    def ends_with(self, operand: NodeIndex) -> PyNodeOperation: ...
    def contains(self, operand: NodeIndex) -> PyNodeOperation: ...

class PyEdgeIndexOperand:
    def greater(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def less(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def greater_or_equal(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def less_or_equal(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def equal(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def not_equal(self, operand: EdgeIndex) -> PyEdgeOperation: ...
    def is_in(self, operand: List[EdgeIndex]) -> PyEdgeOperation: ...
    def not_in(self, operand: List[EdgeIndex]) -> PyEdgeOperation: ...

class PyNodeOperand:
    def in_group(self, operand: Group) -> PyNodeOperation: ...
    def has_attribute(self, operand: MedRecordAttribute) -> PyNodeOperation: ...
    def has_outgoing_edge_with(self, operation: PyEdgeOperation) -> PyNodeOperation: ...
    def has_incoming_edge_with(self, operation: PyEdgeOperation) -> PyNodeOperation: ...
    def has_edge_with(self, operation: PyEdgeOperation) -> PyNodeOperation: ...
    def has_neighbor_with(self, operation: PyNodeOperation) -> PyNodeOperation: ...
    def attribute(self, attribute: MedRecordAttribute) -> PyNodeAttributeOperand: ...
    def index(self) -> PyNodeIndexOperand: ...

class PyEdgeOperand:
    def connected_target(self, operand: NodeIndex) -> PyEdgeOperation: ...
    def connected_source(self, operand: NodeIndex) -> PyEdgeOperation: ...
    def connected(self, operand: NodeIndex) -> PyEdgeOperation: ...
    def has_attribute(self, operand: MedRecordAttribute) -> PyEdgeOperation: ...
    def connected_source_with(self, operation: PyNodeOperation) -> PyEdgeOperation: ...
    def connected_target_with(self, operation: PyNodeOperation) -> PyEdgeOperation: ...
    def connected_with(self, operation: PyNodeOperation) -> PyEdgeOperation: ...
    def has_parallel_edges_with(
        self, operation: PyEdgeOperation
    ) -> PyEdgeOperation: ...
    def has_parallel_edges_with_self_comparison(
        self, operation: PyEdgeOperation
    ) -> PyEdgeOperation: ...
    def attribute(self, attribute: MedRecordAttribute) -> PyEdgeAttributeOperand: ...
    def index(self) -> PyEdgeIndexOperand: ...

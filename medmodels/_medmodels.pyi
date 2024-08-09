from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Union

from medmodels.medrecord.types import (
    Attributes,
    AttributesInput,
    EdgeIndex,
    EdgeIndexInputList,
    EdgeTuple,
    Group,
    GroupInputList,
    MedRecordAttribute,
    MedRecordValue,
    NodeIndex,
    NodeIndexInputList,
    NodeTuple,
    PolarsEdgeDataFrameInput,
    PolarsNodeDataFrameInput,
)

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

PyDataType: TypeAlias = Union[
    PyString,
    PyInt,
    PyFloat,
    PyBool,
    PyDateTime,
    PyNull,
    PyAny,
    PyUnion,
    PyOption,
]

class PyString: ...
class PyInt: ...
class PyFloat: ...
class PyBool: ...
class PyDateTime: ...
class PyNull: ...
class PyAny: ...

class PyUnion:
    dtype1: PyDataType
    dtype2: PyDataType

    def __init__(self, dtype1: PyDataType, dtype2: PyDataType) -> None: ...

class PyOption:
    dtype: PyDataType

    def __init__(self, dtype: PyDataType) -> None: ...

class PyAttributeType(Enum):
    Categorical = ...
    Continuous = ...
    Temporal = ...

class PyAttributeDataType:
    data_type: PyDataType
    attribute_type: Optional[PyAttributeType]

    def __init__(
        self, data_type: PyDataType, attribute_type: Optional[PyAttributeType]
    ) -> None: ...

class PyGroupSchema:
    nodes: Dict[MedRecordAttribute, PyAttributeDataType]
    edges: Dict[MedRecordAttribute, PyAttributeDataType]
    strict: Optional[bool]

    def __init__(
        self,
        *,
        nodes: Dict[MedRecordAttribute, PyAttributeDataType],
        edges: Dict[MedRecordAttribute, PyAttributeDataType],
        strict: Optional[bool] = None,
    ) -> None: ...

class PySchema:
    groups: List[Group]
    default: Optional[PyGroupSchema]
    strict: Optional[bool]

    def __init__(
        self,
        *,
        groups: Dict[Group, PyGroupSchema],
        default: Optional[PyGroupSchema] = None,
        strict: Optional[bool] = None,
    ) -> None: ...
    def group(self, group: Group) -> PyGroupSchema: ...

class PyMedRecord:
    schema: PySchema
    nodes: List[NodeIndex]
    edges: List[EdgeIndex]
    groups: List[Group]

    def __init__(self) -> None: ...
    @staticmethod
    def with_schema(schema: PySchema) -> PyMedRecord: ...
    @staticmethod
    def from_tuples(
        nodes: Sequence[NodeTuple],
        edges: Optional[Sequence[EdgeTuple]],
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
    def from_example_dataset() -> PyMedRecord: ...
    @staticmethod
    def from_ron(path: str) -> PyMedRecord: ...
    def to_ron(self, path: str) -> None: ...
    def update_schema(self, schema: PySchema) -> None: ...
    def node(self, node_index: NodeIndexInputList) -> Dict[NodeIndex, Attributes]: ...
    def edge(self, edge_index: EdgeIndexInputList) -> Dict[EdgeIndex, Attributes]: ...
    def outgoing_edges(
        self, node_index: NodeIndexInputList
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...
    def incoming_edges(
        self, node_index: NodeIndexInputList
    ) -> Dict[NodeIndex, List[EdgeIndex]]: ...
    def edge_endpoints(
        self, edge_index: EdgeIndexInputList
    ) -> Dict[EdgeIndex, tuple[NodeIndex, NodeIndex]]: ...
    def edges_connecting(
        self,
        source_node_indices: NodeIndexInputList,
        target_node_indices: NodeIndexInputList,
    ) -> List[EdgeIndex]: ...
    def edges_connecting_undirected(
        self,
        source_node_indices: NodeIndexInputList,
        target_node_indices: NodeIndexInputList,
    ) -> List[EdgeIndex]: ...
    def add_node(self, node_index: NodeIndex, attributes: AttributesInput) -> None: ...
    def remove_node(
        self, node_index: NodeIndexInputList
    ) -> Dict[NodeIndex, Attributes]: ...
    def replace_node_attributes(
        self, node_index: NodeIndexInputList, attributes: AttributesInput
    ) -> None: ...
    def update_node_attribute(
        self,
        node_index: NodeIndexInputList,
        attribute: MedRecordAttribute,
        value: MedRecordValue,
    ) -> None: ...
    def remove_node_attribute(
        self, node_index: NodeIndexInputList, attribute: MedRecordAttribute
    ) -> None: ...
    def add_nodes(self, nodes: Sequence[NodeTuple]) -> None: ...
    def add_nodes_dataframes(
        self, nodes_dataframe: List[PolarsNodeDataFrameInput]
    ) -> None: ...
    def add_edge(
        self,
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
        attributes: AttributesInput,
    ) -> EdgeIndex: ...
    def remove_edge(
        self, edge_index: EdgeIndexInputList
    ) -> Dict[EdgeIndex, Attributes]: ...
    def replace_edge_attributes(
        self, edge_index: EdgeIndexInputList, attributes: AttributesInput
    ) -> None: ...
    def update_edge_attribute(
        self,
        edge_index: EdgeIndexInputList,
        attribute: MedRecordAttribute,
        value: MedRecordValue,
    ) -> None: ...
    def remove_edge_attribute(
        self, edge_index: EdgeIndexInputList, attribute: MedRecordAttribute
    ) -> None: ...
    def add_edges(self, edges: Sequence[EdgeTuple]) -> List[EdgeIndex]: ...
    def add_edges_dataframes(
        self, edges_dataframe: List[PolarsEdgeDataFrameInput]
    ) -> List[EdgeIndex]: ...
    def add_group(
        self,
        group: Group,
        node_indices_to_add: Optional[NodeIndexInputList],
        edge_indices_to_add: Optional[EdgeIndexInputList],
    ) -> None: ...
    def remove_group(self, group: GroupInputList) -> None: ...
    def add_node_to_group(
        self, group: Group, node_index: NodeIndexInputList
    ) -> None: ...
    def add_edge_to_group(
        self, group: Group, edge_index: EdgeIndexInputList
    ) -> None: ...
    def remove_node_from_group(
        self, group: Group, node_index: NodeIndexInputList
    ) -> None: ...
    def remove_edge_from_group(
        self, group: Group, edge_index: EdgeIndexInputList
    ) -> None: ...
    def nodes_in_group(self, group: GroupInputList) -> Dict[Group, List[NodeIndex]]: ...
    def edges_in_group(self, group: GroupInputList) -> Dict[Group, List[EdgeIndex]]: ...
    def groups_of_node(
        self, node_index: NodeIndexInputList
    ) -> Dict[NodeIndex, List[Group]]: ...
    def groups_of_edge(
        self, edge_index: EdgeIndexInputList
    ) -> Dict[EdgeIndex, List[Group]]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def group_count(self) -> int: ...
    def contains_node(self, node_index: NodeIndex) -> bool: ...
    def contains_edge(self, edge_index: EdgeIndex) -> bool: ...
    def contains_group(self, group: Group) -> bool: ...
    def neighbors(
        self, node_indices: NodeIndexInputList
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...
    def neighbors_undirected(
        self, node_indices: NodeIndexInputList
    ) -> Dict[NodeIndex, List[NodeIndex]]: ...
    def clear(self) -> None: ...
    def select_nodes(
        self, query: Callable[[PyNodeOperand], None]
    ) -> List[NodeIndex]: ...

PyGroupCardinalityWrapper: TypeAlias = Union[Group, List[Group]]
PyValueOperand: TypeAlias = Union[
    PyNodeValueOperand, PyEdgeValueOperand, MedRecordValue
]
PyValuesOperand: TypeAlias = Union[
    PyNodeValuesOperand, PyEdgeValuesOperand, List[MedRecordValue]
]
PyComparisonOperand: TypeAlias = Union[PyValueOperand, PyValuesOperand]

class PyNodeOperand:
    def in_group(self, group: PyGroupCardinalityWrapper) -> None: ...
    def outgoing_edges(self) -> PyEdgeOperand: ...

class PyEdgeOperand:
    def connects_to(self, query: Callable[[PyNodeOperand], None]) -> None: ...
    def attribute(self, attribute: MedRecordAttribute) -> PyEdgeValuesOperand: ...

class PyNodeValuesOperand: ...

class PyEdgeValuesOperand:
    def max(self) -> PyEdgeValueOperand: ...

class PyNodeValueOperand: ...

class PyEdgeValueOperand:
    def less_than(self, value: PyComparisonOperand) -> None: ...

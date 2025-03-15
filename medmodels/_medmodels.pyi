from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, TypeAlias, Union

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

PyDataType: TypeAlias = Union[
    PyString,
    PyInt,
    PyFloat,
    PyBool,
    PyDateTime,
    PyDuration,
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
class PyDuration: ...
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
    Unstructured = ...

    @staticmethod
    def infer(data_type: PyDataType) -> PyAttributeType: ...

class PyAttributeDataType:
    data_type: PyDataType
    attribute_type: PyAttributeType

    def __init__(
        self, data_type: PyDataType, attribute_type: PyAttributeType
    ) -> None: ...

class PyGroupSchema:
    nodes: Dict[MedRecordAttribute, PyAttributeDataType]
    edges: Dict[MedRecordAttribute, PyAttributeDataType]

    def __init__(
        self,
        *,
        nodes: Dict[MedRecordAttribute, PyAttributeDataType],
        edges: Dict[MedRecordAttribute, PyAttributeDataType],
    ) -> None: ...
    def validate_node(self, index: NodeIndex, attributes: Attributes) -> None: ...
    def validate_edge(self, index: EdgeIndex, attributes: Attributes) -> None: ...

class PySchemaType(Enum):
    Provided = ...
    Inferred = ...

class PySchema:
    groups: List[Group]
    default: PyGroupSchema
    schema_type: PySchemaType

    def __init__(
        self,
        *,
        groups: Dict[Group, PyGroupSchema],
        default: PyGroupSchema,
        schema_type: PySchemaType = ...,
    ) -> None: ...
    @staticmethod
    def infer(medrecord: PyMedRecord) -> PySchema: ...
    def group(self, group: Group) -> PyGroupSchema: ...
    def validate_node(
        self, index: NodeIndex, attributes: Attributes, group: Optional[Group] = None
    ) -> None: ...
    def validate_edge(
        self, index: EdgeIndex, attributes: Attributes, group: Optional[Group] = None
    ) -> None: ...
    def set_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: PyDataType,
        attribute_type: PyAttributeType,
        group: Optional[Group] = None,
    ) -> None: ...
    def set_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: PyDataType,
        attribute_type: PyAttributeType,
        group: Optional[Group] = None,
    ) -> None: ...
    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: PyDataType,
        attribute_type: PyAttributeType,
        group: Optional[Group] = None,
    ) -> None: ...
    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: PyDataType,
        attribute_type: PyAttributeType,
        group: Optional[Group] = None,
    ) -> None: ...
    def remove_node_attribute(
        self, attribute: MedRecordAttribute, group: Optional[Group] = None
    ) -> None: ...
    def remove_edge_attribute(
        self, attribute: MedRecordAttribute, group: Optional[Group] = None
    ) -> None: ...
    def add_group(self, group: Group, schema: PyGroupSchema) -> None: ...
    def remove_group(self, group: Group) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...

class PyMedRecord:
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
    def from_simple_example_dataset() -> PyMedRecord: ...
    @staticmethod
    def from_advanced_example_dataset() -> PyMedRecord: ...
    @staticmethod
    def from_ron(path: str) -> PyMedRecord: ...
    def to_ron(self, path: str) -> None: ...
    def get_schema(self) -> PySchema: ...
    def set_schema(self, schema: PySchema) -> None: ...
    def freeze_schema(self) -> None: ...
    def unfreeze_schema(self) -> None: ...
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
    def remove_nodes(
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
    def remove_edges(
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
    def remove_groups(self, group: GroupInputList) -> None: ...
    def add_nodes_to_group(
        self, group: Group, node_index: NodeIndexInputList
    ) -> None: ...
    def add_edges_to_group(
        self, group: Group, edge_index: EdgeIndexInputList
    ) -> None: ...
    def remove_nodes_from_group(
        self, group: Group, node_index: NodeIndexInputList
    ) -> None: ...
    def remove_edges_from_group(
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
    def select_edges(
        self, query: Callable[[PyEdgeOperand], None]
    ) -> List[EdgeIndex]: ...
    def clone(self) -> PyMedRecord: ...

class PyEdgeDirection(Enum):
    Incoming = ...
    Outgoing = ...
    Both = ...

class PyNodeOperand:
    def attribute(self, attribute: MedRecordAttribute) -> PyMultipleValuesOperand: ...
    def attributes(self) -> PyAttributesTreeOperand: ...
    def index(self) -> PyNodeIndicesOperand: ...
    def in_group(self, group: Union[Group, List[Group]]) -> None: ...
    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None: ...
    def edges(self, direction: PyEdgeDirection) -> PyEdgeOperand: ...
    def neighbors(self, direction: PyEdgeDirection) -> PyNodeOperand: ...
    def either_or(
        self,
        either: Callable[[PyNodeOperand], None],
        or_: Callable[[PyNodeOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyNodeOperand], None]) -> None: ...
    def deep_clone(self) -> PyNodeOperand: ...

PyNodeIndexComparisonOperand: TypeAlias = Union[NodeIndex, PyNodeIndexOperand]
PyNodeIndexArithmeticOperand: TypeAlias = PyNodeIndexComparisonOperand
PyNodeIndicesComparisonOperand: TypeAlias = Union[List[NodeIndex], PyNodeIndicesOperand]

class PyNodeIndicesOperand:
    def max(self) -> PyNodeIndexOperand: ...
    def min(self) -> PyNodeIndexOperand: ...
    def count(self) -> PyNodeIndexOperand: ...
    def sum(self) -> PyNodeIndexOperand: ...
    def first(self) -> PyNodeIndexOperand: ...
    def last(self) -> PyNodeIndexOperand: ...
    def greater_than(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def less_than(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def starts_with(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def ends_with(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def contains(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def is_in(self, indices: PyNodeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, indices: PyNodeIndicesComparisonOperand) -> None: ...
    def add(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def sub(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def mul(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def pow(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def mod(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def abs(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyNodeIndicesOperand], None],
        or_: Callable[[PyNodeIndicesOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyNodeIndicesOperand], None]) -> None: ...
    def deep_clone(self) -> PyNodeIndicesOperand: ...

class PyNodeIndexOperand:
    def greater_than(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def less_than(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def starts_with(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def ends_with(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def contains(self, index: PyNodeIndexComparisonOperand) -> None: ...
    def is_in(self, indices: PyNodeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, indices: PyNodeIndicesComparisonOperand) -> None: ...
    def add(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def sub(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def mul(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def pow(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def mod(self, index: PyNodeIndexArithmeticOperand) -> None: ...
    def abs(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyNodeIndexOperand], None],
        or_: Callable[[PyNodeIndexOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyNodeIndexOperand], None]) -> None: ...
    def deep_clone(self) -> PyNodeIndexOperand: ...

class PyEdgeOperand:
    def attribute(self, attribute: MedRecordAttribute) -> PyMultipleValuesOperand: ...
    def attributes(self) -> PyAttributesTreeOperand: ...
    def index(self) -> PyEdgeIndicesOperand: ...
    def in_group(self, group: Union[Group, List[Group]]) -> None: ...
    def has_attribute(
        self, attribute: Union[MedRecordAttribute, List[MedRecordAttribute]]
    ) -> None: ...
    def source_node(self) -> PyNodeOperand: ...
    def target_node(self) -> PyNodeOperand: ...
    def either_or(
        self,
        either: Callable[[PyEdgeOperand], None],
        or_: Callable[[PyEdgeOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyEdgeOperand], None]) -> None: ...
    def deep_clone(self) -> PyEdgeOperand: ...

PyEdgeIndexComparisonOperand: TypeAlias = Union[EdgeIndex, PyEdgeIndexOperand]
PyEdgeIndexArithmeticOperand: TypeAlias = PyEdgeIndexComparisonOperand
PyEdgeIndicesComparisonOperand: TypeAlias = Union[List[EdgeIndex], PyEdgeIndicesOperand]

class PyEdgeIndicesOperand:
    def max(self) -> PyEdgeIndexOperand: ...
    def min(self) -> PyEdgeIndexOperand: ...
    def count(self) -> PyEdgeIndexOperand: ...
    def sum(self) -> PyEdgeIndexOperand: ...
    def first(self) -> PyEdgeIndexOperand: ...
    def last(self) -> PyEdgeIndexOperand: ...
    def greater_than(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def less_than(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def starts_with(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def ends_with(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def contains(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def is_in(self, indices: PyEdgeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, indices: PyEdgeIndicesComparisonOperand) -> None: ...
    def add(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def sub(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def mul(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def pow(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def mod(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyEdgeIndicesOperand], None],
        or_: Callable[[PyEdgeIndicesOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyEdgeIndicesOperand], None]) -> None: ...
    def deep_clone(self) -> PyEdgeIndicesOperand: ...

class PyEdgeIndexOperand:
    def greater_than(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def less_than(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def starts_with(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def ends_with(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def contains(self, index: PyEdgeIndexComparisonOperand) -> None: ...
    def is_in(self, indices: PyEdgeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, indices: PyEdgeIndicesComparisonOperand) -> None: ...
    def add(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def sub(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def mul(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def pow(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def mod(self, index: PyEdgeIndexArithmeticOperand) -> None: ...
    def either_or(
        self,
        either: Callable[[PyEdgeIndexOperand], None],
        or_: Callable[[PyEdgeIndexOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyEdgeIndexOperand], None]) -> None: ...
    def deep_clone(self) -> PyEdgeIndexOperand: ...

PySingleValueComparisonOperand: TypeAlias = Union[MedRecordValue, PySingleValueOperand]
PySingleValueArithmeticOperand: TypeAlias = PySingleValueComparisonOperand
PyMultipleValuesComparisonOperand: TypeAlias = Union[
    List[MedRecordValue], PyMultipleValuesOperand
]

class PyMultipleValuesOperand:
    def max(self) -> PySingleValueOperand: ...
    def min(self) -> PySingleValueOperand: ...
    def mean(self) -> PySingleValueOperand: ...
    def median(self) -> PySingleValueOperand: ...
    def mode(self) -> PySingleValueOperand: ...
    def std(self) -> PySingleValueOperand: ...
    def var(self) -> PySingleValueOperand: ...
    def count(self) -> PySingleValueOperand: ...
    def sum(self) -> PySingleValueOperand: ...
    def first(self) -> PySingleValueOperand: ...
    def last(self) -> PySingleValueOperand: ...
    def greater_than(self, value: PySingleValueComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, value: PySingleValueComparisonOperand
    ) -> None: ...
    def less_than(self, value: PySingleValueComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def not_equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def starts_with(self, value: PySingleValueComparisonOperand) -> None: ...
    def ends_with(self, value: PySingleValueComparisonOperand) -> None: ...
    def contains(self, value: PySingleValueComparisonOperand) -> None: ...
    def is_in(self, values: PyMultipleValuesComparisonOperand) -> None: ...
    def is_not_in(self, values: PyMultipleValuesComparisonOperand) -> None: ...
    def add(self, value: PySingleValueArithmeticOperand) -> None: ...
    def sub(self, value: PySingleValueArithmeticOperand) -> None: ...
    def mul(self, value: PySingleValueArithmeticOperand) -> None: ...
    def div(self, value: PySingleValueArithmeticOperand) -> None: ...
    def pow(self, value: PySingleValueArithmeticOperand) -> None: ...
    def mod(self, value: PySingleValueArithmeticOperand) -> None: ...
    def round(self) -> None: ...
    def ceil(self) -> None: ...
    def floor(self) -> None: ...
    def abs(self) -> None: ...
    def sqrt(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_float(self) -> None: ...
    def is_bool(self) -> None: ...
    def is_datetime(self) -> None: ...
    def is_duration(self) -> None: ...
    def is_null(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyMultipleValuesOperand], None],
        or_: Callable[[PyMultipleValuesOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyMultipleValuesOperand], None]) -> None: ...
    def deep_clone(self) -> PyMultipleValuesOperand: ...

class PySingleValueOperand:
    def greater_than(self, value: PySingleValueComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, value: PySingleValueComparisonOperand
    ) -> None: ...
    def less_than(self, value: PySingleValueComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def not_equal_to(self, value: PySingleValueComparisonOperand) -> None: ...
    def starts_with(self, value: PySingleValueComparisonOperand) -> None: ...
    def ends_with(self, value: PySingleValueComparisonOperand) -> None: ...
    def contains(self, value: PySingleValueComparisonOperand) -> None: ...
    def is_in(self, values: PyMultipleValuesComparisonOperand) -> None: ...
    def is_not_in(self, values: PyMultipleValuesComparisonOperand) -> None: ...
    def add(self, value: PySingleValueArithmeticOperand) -> None: ...
    def sub(self, value: PySingleValueArithmeticOperand) -> None: ...
    def mul(self, value: PySingleValueArithmeticOperand) -> None: ...
    def div(self, value: PySingleValueArithmeticOperand) -> None: ...
    def pow(self, value: PySingleValueArithmeticOperand) -> None: ...
    def mod(self, value: PySingleValueArithmeticOperand) -> None: ...
    def round(self) -> None: ...
    def ceil(self) -> None: ...
    def floor(self) -> None: ...
    def abs(self) -> None: ...
    def sqrt(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_float(self) -> None: ...
    def is_bool(self) -> None: ...
    def is_datetime(self) -> None: ...
    def is_duration(self) -> None: ...
    def is_null(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PySingleValueOperand], None],
        or_: Callable[[PySingleValueOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PySingleValueOperand], None]) -> None: ...
    def deep_clone(self) -> PySingleValueOperand: ...

PySingleAttributeComparisonOperand: TypeAlias = Union[
    MedRecordAttribute, PySingleAttributeOperand
]
PySingleAttributeArithmeticOperand: TypeAlias = PySingleAttributeComparisonOperand
PyMultipleAttributesComparisonOperand: TypeAlias = Union[
    List[MedRecordAttribute], PyMultipleAttributesOperand
]

class PyAttributesTreeOperand:
    def max(self) -> PyMultipleAttributesOperand: ...
    def min(self) -> PyMultipleAttributesOperand: ...
    def count(self) -> PyMultipleAttributesOperand: ...
    def sum(self) -> PyMultipleAttributesOperand: ...
    def first(self) -> PyMultipleAttributesOperand: ...
    def last(self) -> PyMultipleAttributesOperand: ...
    def greater_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def starts_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def contains(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def is_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def add(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def sub(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mul(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def pow(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mod(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def abs(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyAttributesTreeOperand], None],
        or_: Callable[[PyAttributesTreeOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyAttributesTreeOperand], None]) -> None: ...
    def deep_clone(self) -> PyAttributesTreeOperand: ...

class PyMultipleAttributesOperand:
    def max(self) -> PySingleAttributeOperand: ...
    def min(self) -> PySingleAttributeOperand: ...
    def count(self) -> PySingleAttributeOperand: ...
    def sum(self) -> PySingleAttributeOperand: ...
    def first(self) -> PySingleAttributeOperand: ...
    def last(self) -> PySingleAttributeOperand: ...
    def greater_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def starts_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def contains(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def is_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def add(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def sub(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mul(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def pow(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mod(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def abs(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def to_values(self) -> PyMultipleValuesOperand: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PyMultipleAttributesOperand], None],
        or_: Callable[[PyMultipleAttributesOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PyMultipleAttributesOperand], None]) -> None: ...
    def deep_clone(self) -> PyMultipleAttributesOperand: ...

class PySingleAttributeOperand:
    def greater_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, attribute: PySingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def starts_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def contains(self, attribute: PySingleAttributeComparisonOperand) -> None: ...
    def is_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, attributes: PyMultipleAttributesComparisonOperand) -> None: ...
    def add(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def sub(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mul(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def pow(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def mod(self, attribute: PySingleAttributeArithmeticOperand) -> None: ...
    def abs(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def either_or(
        self,
        either: Callable[[PySingleAttributeOperand], None],
        or_: Callable[[PySingleAttributeOperand], None],
    ) -> None: ...
    def exclude(self, query: Callable[[PySingleAttributeOperand], None]) -> None: ...
    def deep_clone(self) -> PySingleAttributeOperand: ...

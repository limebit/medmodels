from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Callable, List, Union

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

NodeQuery: TypeAlias = Callable[[NodeOperand], None]
EdgeQuery: TypeAlias = Callable[[EdgeOperand], None]

SingleValueComparisonOperand: TypeAlias = Union[SingleValueOperand, MedRecordValue]
MultipleValuesComparisonOperand: TypeAlias = Union[
    MultipleValuesOperand, List[MedRecordValue]
]

SingleAttributeComparisonOperand: TypeAlias = Union[
    SingleAttributeOperand,
    MedRecordAttribute,
]
MultipleAttributesComparisonOperand: TypeAlias = Union[
    MultipleAttributesOperand, List[MedRecordAttribute]
]

NodeIndexComparisonOperand: TypeAlias = Union[NodeIndexOperand, NodeIndex]
NodeIndicesComparisonOperand: TypeAlias = Union[NodeIndicesOperand, List[NodeIndex]]

EdgeIndexComparisonOperand: TypeAlias = Union[
    EdgeIndexOperand,
    EdgeIndex,
]
EdgeIndicesComparisonOperand: TypeAlias = Union[
    EdgeIndicesOperand,
    List[EdgeIndex],
]

class EdgeDirection(Enum):
    INCOMING = auto()
    OUTGOING = auto()
    BOTH = auto()

class NodeOperand:
    def attribute(
        self, attribute: Union[MedRecordAttribute, SingleAttributeOperand]
    ) -> MultipleValuesOperand: ...
    def attributes(self) -> MultipleAttributesOperand: ...
    def index(self) -> NodeIndexOperand: ...
    def in_group(self, group: Union[Group, List[Group]]) -> None: ...
    def has_attribute(
        self, attribute: Union[MedRecordAttribute, SingleAttributeOperand]
    ) -> None: ...
    def incoming_edges(self) -> EdgeOperand: ...
    def outgoing_edges(self) -> EdgeOperand: ...
    def neighbors(
        self, edge_direction: EdgeDirection = EdgeDirection.OUTGOING
    ) -> NodeOperand: ...
    def either_or(self, either: NodeQuery, or_: NodeQuery) -> None: ...
    def clone(self) -> NodeOperand: ...

class EdgeOperand:
    def attribute(
        self, attribute: Union[MedRecordAttribute, SingleAttributeOperand]
    ) -> MultipleValuesOperand: ...
    def attributes(self) -> MultipleAttributesOperand: ...
    def index(self) -> EdgeIndexOperand: ...
    def in_group(self, group: Union[Group, List[Group]]) -> None: ...
    def has_attribute(
        self, attribute: Union[MedRecordAttribute, SingleAttributeOperand]
    ) -> None: ...
    def source_node(self) -> NodeOperand: ...
    def target_node(self) -> NodeOperand: ...
    def either_or(self, either: EdgeQuery, or_: EdgeQuery) -> None: ...
    def clone(self) -> EdgeOperand: ...

class MultipleValuesOperand:
    def max(self) -> SingleValueOperand: ...
    def min(self) -> SingleValueOperand: ...
    def mean(self) -> SingleValueOperand: ...
    def median(self) -> SingleValueOperand: ...
    def mode(self) -> SingleValueOperand: ...
    def std(self) -> SingleValueOperand: ...
    def var(self) -> SingleValueOperand: ...
    def count(self) -> SingleValueOperand: ...
    def sum(self) -> SingleValueOperand: ...
    def first(self) -> SingleValueOperand: ...
    def last(self) -> SingleValueOperand: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_float(self) -> None: ...
    def is_bool(self) -> None: ...
    def is_datetime(self) -> None: ...
    def is_null(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def greater_than(self, value: SingleValueComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def less_than(self, value: SingleValueComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def not_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def is_in(self, values: MultipleValuesComparisonOperand) -> None: ...
    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None: ...
    def starts_with(self, value: SingleValueComparisonOperand) -> None: ...
    def ends_with(self, value: SingleValueComparisonOperand) -> None: ...
    def contains(self, value: SingleValueComparisonOperand) -> None: ...
    def add(self, value: SingleValueComparisonOperand) -> None: ...
    def subtract(self, value: SingleValueComparisonOperand) -> None: ...
    def multiply(self, value: SingleValueComparisonOperand) -> None: ...
    def divide(self, value: SingleValueComparisonOperand) -> None: ...
    def modulo(self, value: SingleValueComparisonOperand) -> None: ...
    def power(self, value: SingleValueComparisonOperand) -> None: ...
    def round(self) -> None: ...
    def ceil(self) -> None: ...
    def floor(self) -> None: ...
    def absolute(self) -> None: ...
    def sqrt(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[MultipleValuesOperand], None],
        or_: Callable[[MultipleValuesOperand], None],
    ) -> None: ...
    def clone(self) -> MultipleValuesOperand: ...

class SingleValueOperand:
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_float(self) -> None: ...
    def is_bool(self) -> None: ...
    def is_datetime(self) -> None: ...
    def is_null(self) -> None: ...
    def greater_than(self, value: SingleValueComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def less_than(self, value: SingleValueComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def not_equal_to(self, value: SingleValueComparisonOperand) -> None: ...
    def is_in(self, values: MultipleValuesComparisonOperand) -> None: ...
    def is_not_in(self, values: MultipleValuesComparisonOperand) -> None: ...
    def starts_with(self, value: SingleValueComparisonOperand) -> None: ...
    def ends_with(self, value: SingleValueComparisonOperand) -> None: ...
    def contains(self, value: SingleValueComparisonOperand) -> None: ...
    def add(self, value: SingleValueComparisonOperand) -> None: ...
    def subtract(self, value: SingleValueComparisonOperand) -> None: ...
    def multiply(self, value: SingleValueComparisonOperand) -> None: ...
    def modulo(self, value: SingleValueComparisonOperand) -> None: ...
    def power(self, value: SingleValueComparisonOperand) -> None: ...
    def round(self) -> None: ...
    def ceil(self) -> None: ...
    def floor(self) -> None: ...
    def absolute(self) -> None: ...
    def sqrt(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[SingleValueOperand], None],
        or_: Callable[[SingleValueOperand], None],
    ) -> None: ...
    def clone(self) -> SingleValueOperand: ...

class AttributesTreeOperand:
    def max(self) -> MultipleAttributesOperand: ...
    def min(self) -> MultipleAttributesOperand: ...
    def count(self) -> MultipleAttributesOperand: ...
    def sum(self) -> MultipleAttributesOperand: ...
    def first(self) -> MultipleAttributesOperand: ...
    def last(self) -> MultipleAttributesOperand: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def greater_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def is_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def starts_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def contains(self, value: SingleAttributeComparisonOperand) -> None: ...
    def add(self, value: SingleAttributeComparisonOperand) -> None: ...
    def subtract(self, value: SingleAttributeComparisonOperand) -> None: ...
    def multiply(self, value: SingleAttributeComparisonOperand) -> None: ...
    def modulo(self, value: SingleAttributeComparisonOperand) -> None: ...
    def power(self, value: SingleAttributeComparisonOperand) -> None: ...
    def absolute(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[AttributesTreeOperand], None],
        or_: Callable[[AttributesTreeOperand], None],
    ) -> None: ...
    def clone(self) -> AttributesTreeOperand: ...

class MultipleAttributesOperand:
    def max(self) -> SingleAttributeOperand: ...
    def min(self) -> SingleAttributeOperand: ...
    def count(self) -> SingleAttributeOperand: ...
    def sum(self) -> SingleAttributeOperand: ...
    def first(self) -> SingleAttributeOperand: ...
    def last(self) -> SingleAttributeOperand: ...
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def is_max(self) -> None: ...
    def is_min(self) -> None: ...
    def greater_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def is_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def starts_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def contains(self, value: SingleAttributeComparisonOperand) -> None: ...
    def add(self, value: SingleAttributeComparisonOperand) -> None: ...
    def subtract(self, value: SingleAttributeComparisonOperand) -> None: ...
    def multiply(self, value: SingleAttributeComparisonOperand) -> None: ...
    def modulo(self, value: SingleAttributeComparisonOperand) -> None: ...
    def power(self, value: SingleAttributeComparisonOperand) -> None: ...
    def absolute(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def to_values(self) -> MultipleValuesOperand: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[MultipleAttributesOperand], None],
        or_: Callable[[MultipleAttributesOperand], None],
    ) -> None: ...
    def clone(self) -> MultipleAttributesOperand: ...

class SingleAttributeOperand:
    def is_string(self) -> None: ...
    def is_int(self) -> None: ...
    def greater_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def greater_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def less_than(self, value: SingleAttributeComparisonOperand) -> None: ...
    def less_than_or_equal_to(
        self, value: SingleAttributeComparisonOperand
    ) -> None: ...
    def equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def not_equal_to(self, value: SingleAttributeComparisonOperand) -> None: ...
    def is_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def is_not_in(self, values: MultipleAttributesComparisonOperand) -> None: ...
    def starts_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def ends_with(self, value: SingleAttributeComparisonOperand) -> None: ...
    def contains(self, value: SingleAttributeComparisonOperand) -> None: ...
    def add(self, value: SingleAttributeComparisonOperand) -> None: ...
    def subtract(self, value: SingleAttributeComparisonOperand) -> None: ...
    def multiply(self, value: SingleAttributeComparisonOperand) -> None: ...
    def modulo(self, value: SingleAttributeComparisonOperand) -> None: ...
    def power(self, value: SingleAttributeComparisonOperand) -> None: ...
    def absolute(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[SingleAttributeOperand], None],
        or_: Callable[[SingleAttributeOperand], None],
    ) -> None: ...
    def clone(self) -> SingleAttributeOperand: ...

class NodeIndicesOperand:
    def max(self) -> NodeIndexOperand: ...
    def min(self) -> NodeIndexOperand: ...
    def count(self) -> NodeIndexOperand: ...
    def sum(self) -> NodeIndexOperand: ...
    def first(self) -> NodeIndexOperand: ...
    def last(self) -> NodeIndexOperand: ...
    def greater_than(self, value: NodeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def less_than(self, value: NodeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def is_in(self, values: NodeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, values: NodeIndicesComparisonOperand) -> None: ...
    def starts_with(self, value: NodeIndexComparisonOperand) -> None: ...
    def ends_with(self, value: NodeIndexComparisonOperand) -> None: ...
    def contains(self, value: NodeIndexComparisonOperand) -> None: ...
    def add(self, value: NodeIndexComparisonOperand) -> None: ...
    def subtract(self, value: NodeIndexComparisonOperand) -> None: ...
    def multiply(self, value: NodeIndexComparisonOperand) -> None: ...
    def modulo(self, value: NodeIndexComparisonOperand) -> None: ...
    def power(self, value: NodeIndexComparisonOperand) -> None: ...
    def absolute(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[NodeIndicesOperand], None],
        or_: Callable[[NodeIndicesOperand], None],
    ) -> None: ...
    def clone(self) -> NodeIndicesOperand: ...

class NodeIndexOperand:
    def greater_than(self, value: NodeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def less_than(self, value: NodeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, value: NodeIndexComparisonOperand) -> None: ...
    def is_in(self, values: NodeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, values: NodeIndicesComparisonOperand) -> None: ...
    def starts_with(self, value: NodeIndexComparisonOperand) -> None: ...
    def ends_with(self, value: NodeIndexComparisonOperand) -> None: ...
    def contains(self, value: NodeIndexComparisonOperand) -> None: ...
    def add(self, value: NodeIndexComparisonOperand) -> None: ...
    def subtract(self, value: NodeIndexComparisonOperand) -> None: ...
    def multiply(self, value: NodeIndexComparisonOperand) -> None: ...
    def modulo(self, value: NodeIndexComparisonOperand) -> None: ...
    def power(self, value: NodeIndexComparisonOperand) -> None: ...
    def absolute(self) -> None: ...
    def trim(self) -> None: ...
    def trim_start(self) -> None: ...
    def trim_end(self) -> None: ...
    def lowercase(self) -> None: ...
    def uppercase(self) -> None: ...
    def slice(self, start: int, end: int) -> None: ...
    def either_or(
        self,
        either: Callable[[NodeIndexOperand], None],
        or_: Callable[[NodeIndexOperand], None],
    ) -> None: ...
    def clone(self) -> NodeIndexOperand: ...

class EdgeIndicesOperand:
    def max(self) -> EdgeIndexOperand: ...
    def min(self) -> EdgeIndexOperand: ...
    def count(self) -> EdgeIndexOperand: ...
    def sum(self) -> EdgeIndexOperand: ...
    def first(self) -> EdgeIndexOperand: ...
    def last(self) -> EdgeIndexOperand: ...
    def greater_than(self, value: EdgeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def less_than(self, value: EdgeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def is_in(self, values: EdgeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, values: EdgeIndicesComparisonOperand) -> None: ...
    def starts_with(self, value: EdgeIndexComparisonOperand) -> None: ...
    def ends_with(self, value: EdgeIndexComparisonOperand) -> None: ...
    def contains(self, value: EdgeIndexComparisonOperand) -> None: ...
    def add(self, value: EdgeIndexComparisonOperand) -> None: ...
    def subtract(self, value: EdgeIndexComparisonOperand) -> None: ...
    def multiply(self, value: EdgeIndexComparisonOperand) -> None: ...
    def modulo(self, value: EdgeIndexComparisonOperand) -> None: ...
    def power(self, value: EdgeIndexComparisonOperand) -> None: ...
    def either_or(
        self,
        either: Callable[[EdgeIndicesOperand], None],
        or_: Callable[[EdgeIndicesOperand], None],
    ) -> None: ...
    def clone(self) -> EdgeIndicesOperand: ...

class EdgeIndexOperand:
    def greater_than(self, value: EdgeIndexComparisonOperand) -> None: ...
    def greater_than_or_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def less_than(self, value: EdgeIndexComparisonOperand) -> None: ...
    def less_than_or_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def not_equal_to(self, value: EdgeIndexComparisonOperand) -> None: ...
    def is_in(self, values: EdgeIndicesComparisonOperand) -> None: ...
    def is_not_in(self, values: EdgeIndicesComparisonOperand) -> None: ...
    def starts_with(self, value: EdgeIndicesComparisonOperand) -> None: ...
    def ends_with(self, value: EdgeIndicesComparisonOperand) -> None: ...
    def contains(self, value: EdgeIndicesComparisonOperand) -> None: ...
    def add(self, value: EdgeIndexComparisonOperand) -> None: ...
    def subtract(self, value: EdgeIndexComparisonOperand) -> None: ...
    def multiply(self, value: EdgeIndexComparisonOperand) -> None: ...
    def modulo(self, value: EdgeIndexComparisonOperand) -> None: ...
    def power(self, value: EdgeIndexComparisonOperand) -> None: ...
    def either_or(
        self,
        either: Callable[[EdgeIndexOperand], None],
        or_: Callable[[EdgeIndexOperand], None],
    ) -> None: ...
    def clone(self) -> EdgeIndexOperand: ...
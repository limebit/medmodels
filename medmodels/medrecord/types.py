"""Type aliases and type checking functions for medical record data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from typing_extensions import TypeIs

#: A type alias for attributes of a medical record.
MedRecordAttribute: TypeAlias = Union[str, int]

#: A type alias for a list of medical record attributes.
MedRecordAttributeInputList: TypeAlias = Union[
    List[str], List[int], List[MedRecordAttribute]
]

#: A type alias for the value of a medical record attribute.
MedRecordValue: TypeAlias = Union[str, int, float, bool, datetime, timedelta, None]

#: A type alias for a node index.
NodeIndex: TypeAlias = MedRecordAttribute

#: A type alias for a list of node indices.
NodeIndexInputList: TypeAlias = MedRecordAttributeInputList

#: A type alias for an edge index.
EdgeIndex: TypeAlias = int

#: A type alias for a list of edge indices.
EdgeIndexInputList: TypeAlias = List[EdgeIndex]

#: A type alias for a group.
Group: TypeAlias = MedRecordAttribute

#: A type alias for a list of groups.
GroupInputList: TypeAlias = MedRecordAttributeInputList

#: A type alias for attributes.
Attributes: TypeAlias = Dict[MedRecordAttribute, MedRecordValue]

#: A type alias for input attributes.
AttributesInput: TypeAlias = Union[
    Mapping[MedRecordAttribute, MedRecordValue],
    Mapping[str, MedRecordValue],
    Mapping[int, MedRecordValue],
]

#: A type alias for a node tuple.
NodeTuple: TypeAlias = Union[
    Tuple[str, AttributesInput],
    Tuple[int, AttributesInput],
    Tuple[NodeIndex, AttributesInput],
]

#: A type alias for an edge tuple.
EdgeTuple: TypeAlias = Union[
    Tuple[str, str, AttributesInput],
    Tuple[str, int, AttributesInput],
    Tuple[str, NodeIndex, AttributesInput],
    Tuple[int, str, AttributesInput],
    Tuple[int, int, AttributesInput],
    Tuple[int, NodeIndex, AttributesInput],
    Tuple[NodeIndex, str, AttributesInput],
    Tuple[NodeIndex, int, AttributesInput],
    Tuple[NodeIndex, NodeIndex, AttributesInput],
]

#: A type alias for input to a Polars DataFrame for nodes.
PolarsNodeDataFrameInput: TypeAlias = Tuple[pl.DataFrame, str]

#: A type alias for input to a Polars DataFrame for edges.
PolarsEdgeDataFrameInput: TypeAlias = Tuple[pl.DataFrame, str, str]

#: A type alias for input to a Pandas DataFrame for nodes.
PandasNodeDataFrameInput: TypeAlias = Tuple[pd.DataFrame, str]

#: A type alias for input to a Pandas DataFrame for edges.
PandasEdgeDataFrameInput: TypeAlias = Tuple[pd.DataFrame, str, str]

#: A type alias for input to a node.
NodeInput = Union[
    NodeTuple,
    Sequence[NodeTuple],
    PandasNodeDataFrameInput,
    List[PandasNodeDataFrameInput],
    PolarsNodeDataFrameInput,
    List[PolarsNodeDataFrameInput],
]

#: A type alias for input to an edge.
EdgeInput = Union[
    EdgeTuple,
    Sequence[EdgeTuple],
    PandasEdgeDataFrameInput,
    List[PandasEdgeDataFrameInput],
    PolarsEdgeDataFrameInput,
    List[PolarsEdgeDataFrameInput],
]


class GroupInfo(TypedDict):
    """A dictionary containing lists of node and edge indices for a group."""

    nodes: List[NodeIndex]
    edges: List[EdgeIndex]


class AttributeInfo(TypedDict):
    """A dictionary containing info about nodes/edges and their attributes."""

    count: int
    attribute: Dict[
        MedRecordAttribute,
        Union[TemporalAttributeInfo, NumericAttributeInfo, StringAttributeInfo],
    ]


class TemporalAttributeInfo(TypedDict):
    """Dictionary for a temporal attribute and its metrics."""

    type: Literal["Temporal"]
    min: datetime
    max: datetime


class NumericAttributeInfo(TypedDict):
    """Dictionary for a numeric attribute and its metrics."""

    type: Literal["Continuous"]
    min: Union[int, float]
    max: Union[int, float]
    mean: Union[int, float]


class StringAttributeInfo(TypedDict):
    """Dictionary for a string attribute and its values."""

    type: Literal["Categorical"]
    values: str


def is_medrecord_attribute(value: object) -> TypeIs[MedRecordAttribute]:
    """Check if a value is a MedRecord attribute.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[MedRecordAttribute]: True if the value is a MedRecord attribute,
            otherwise False.
    """
    return isinstance(value, (str, int))


def is_medrecord_value(value: object) -> TypeIs[MedRecordValue]:
    """Check if a value is a valid MedRecord value.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[MedRecordValue]: True if the value is a valid MedRecord value, otherwise
            False.
    """
    return isinstance(value, (str, int, float, bool, datetime)) or value is None


def is_node_index(value: object) -> TypeIs[NodeIndex]:
    """Check if a value is a valid node index.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[NodeIndex]: True if the value is a valid node index, otherwise False.
    """
    return is_medrecord_attribute(value)


def is_node_index_list(value: object) -> TypeIs[NodeIndexInputList]:
    """Check if a value is a valid list of node indices.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[NodeIndexInputList]: True if the value is a valid list of node indices,
            otherwise False.
    """
    return isinstance(value, list) and all(is_node_index(input) for input in value)


def is_edge_index(value: object) -> TypeIs[EdgeIndex]:
    """Check if a value is a valid edge index.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[EdgeIndex]: True if the value is a valid edge index, otherwise False.
    """
    return isinstance(value, int)


def is_edge_index_list(value: object) -> TypeIs[EdgeIndexInputList]:
    """Check if a value is a valid list of edge indices.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[EdgeIndexInputList]: True if the value is a valid list of edge indices,
            otherwise False.
    """
    return isinstance(value, list) and all(is_edge_index(input) for input in value)


def is_group(value: object) -> TypeIs[Group]:
    """Check if a value is a valid group.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[Group]: True if the value is a valid group, otherwise False.
    """
    return is_medrecord_attribute(value)


def is_attributes(value: object) -> TypeIs[Attributes]:
    """Check if a value is a valid attributes dictionary.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[Attributes]: True if the value is a valid attributes dictionary,
            otherwise False.
    """
    return isinstance(value, dict)


def is_node_tuple(value: object) -> TypeIs[NodeTuple]:
    """Check if a value is a valid node tuple.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[NodeTuple]: True if the value is a valid node tuple, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and is_medrecord_attribute(value[0])
        and is_attributes(value[1])
    )


def is_node_tuple_list(value: object) -> TypeIs[List[NodeTuple]]:
    """Check if a value is a list of valid node tuples.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[NodeTuple]]: True if the value is a list of valid node tuples,
            otherwise False.
    """
    return isinstance(value, list) and all(is_node_tuple(input) for input in value)


def is_edge_tuple(value: object) -> TypeIs[EdgeTuple]:
    """Check if a value is a valid edge tuple.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[EdgeTuple]: True if the value is a valid edge tuple, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and is_medrecord_attribute(value[0])
        and is_medrecord_attribute(value[1])
        and is_attributes(value[2])
    )


def is_edge_tuple_list(value: object) -> TypeIs[List[EdgeTuple]]:
    """Check if a value is a list of valid edge tuples.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[EdgeTuple]]: True if the value is a list of valid edge tuples,
            otherwise False.
    """
    return isinstance(value, list) and all(is_edge_tuple(input) for input in value)


def is_polars_node_dataframe_input(
    value: object,
) -> TypeIs[PolarsNodeDataFrameInput]:
    """Check if a value is a valid Polars DataFrame input for nodes.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[PolarsNodeDataFrameInput]: True if the value is a valid Polars DataFrame
            input for nodes, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], pl.DataFrame)
        and isinstance(value[1], str)
    )


def is_polars_node_dataframe_input_list(
    value: object,
) -> TypeIs[List[PolarsNodeDataFrameInput]]:
    """Check if a value is a list of valid Polars DataFrame inputs for nodes.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[PolarsNodeDataFrameInput]]: True if the value is a list of valid
            Polars DataFrame inputs for nodes, otherwise False.
    """
    return isinstance(value, list) and all(
        is_polars_node_dataframe_input(input) for input in value
    )


def is_polars_edge_dataframe_input(
    value: object,
) -> TypeIs[PolarsEdgeDataFrameInput]:
    """Check if a value is a valid Polars DataFrame input for edges.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[PolarsEdgeDataFrameInput]: True if the value is a valid Polars DataFrame
            input for edges, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], pl.DataFrame)
        and isinstance(value[1], str)
        and isinstance(value[2], str)
    )


def is_polars_edge_dataframe_input_list(
    value: object,
) -> TypeIs[List[PolarsEdgeDataFrameInput]]:
    """Check if a value is a list of valid Polars DataFrame inputs for edges.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[PolarsEdgeDataFrameInput]]: True if the value is a list of valid
            Polars DataFrame inputs for edges, otherwise False.
    """
    return isinstance(value, list) and all(
        is_polars_edge_dataframe_input(input) for input in value
    )


def is_pandas_node_dataframe_input(
    value: object,
) -> TypeIs[PandasNodeDataFrameInput]:
    """Check if a value is a valid Pandas DataFrame input for nodes.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[PandasNodeDataFrameInput]: True if the value is a valid Pandas DataFrame
            input for nodes, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], pd.DataFrame)
        and isinstance(value[1], str)
    )


def is_pandas_node_dataframe_input_list(
    value: object,
) -> TypeIs[List[PandasNodeDataFrameInput]]:
    """Check if a value is a list of valid Pandas DataFrame inputs for nodes.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[PandasNodeDataFrameInput]]: True if the value is a list of valid
            Pandas DataFrame inputs for nodes, otherwise False.
    """
    return isinstance(value, list) and all(
        is_pandas_node_dataframe_input(input) for input in value
    )


def is_pandas_edge_dataframe_input(
    value: object,
) -> TypeIs[PandasEdgeDataFrameInput]:
    """Check if a value is a valid Pandas DataFrame input for edges.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[PandasEdgeDataFrameInput]: True if the value is a valid Pandas DataFrame
            input for edges, otherwise False.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], pd.DataFrame)
        and isinstance(value[1], str)
        and isinstance(value[2], str)
    )


def is_pandas_edge_dataframe_input_list(
    value: object,
) -> TypeIs[List[PandasEdgeDataFrameInput]]:
    """Check if a value is a list of valid Pandas DataFrame inputs for edges.

    Args:
        value (object): The value to check.

    Returns:
        TypeIs[List[PandasEdgeDataFrameInput]]: True if the value is a list of valid
            Pandas DataFrame inputs for edges, otherwise False.
    """
    return isinstance(value, list) and all(
        is_pandas_edge_dataframe_input(input) for input in value
    )

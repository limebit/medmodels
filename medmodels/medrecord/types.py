from typing import Dict, Union
from polars import DataFrame

MedRecordAttribute = Union[str, int]
MedRecordValue = Union[str, int, float, bool, None]
NodeIndex = MedRecordAttribute
EdgeIndex = int
Group = MedRecordAttribute
Attributes = Dict[MedRecordAttribute, MedRecordValue]
NodeDataFrameInput = tuple[DataFrame, str]
EdgeDataFrameInput = tuple[DataFrame, str, str]


def is_medrecord_attribute(value: MedRecordAttribute) -> bool:
    return isinstance(value, (str, int))


def is_medrecord_value(value: MedRecordValue) -> bool:
    return isinstance(value, (str, int, float, bool, None))


def is_node_index(value: NodeIndex) -> bool:
    return is_medrecord_attribute(value)


def is_edge_index(value: EdgeIndex) -> bool:
    return isinstance(value, int)


def is_group(value: Group) -> bool:
    return is_medrecord_attribute(value)


def is_attributes(value: Attributes) -> bool:
    return isinstance(value, Attributes)


def is_node_dataframe_input(value: NodeDataFrameInput) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], DataFrame)
        and isinstance(value[1], str)
    )


def is_edge_dataframe_input(value: EdgeDataFrameInput) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], DataFrame)
        and isinstance(value[1], str)
        and isinstance(value[2], str)
    )

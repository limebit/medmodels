from typing import Dict, Union

MedRecordAttribute = Union[str, int]
MedRecordValue = Union[str, int, float, bool, None]
NodeIndex = MedRecordAttribute
EdgeIndex = int
Group = MedRecordAttribute
Attributes = Dict[MedRecordAttribute, MedRecordValue]


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

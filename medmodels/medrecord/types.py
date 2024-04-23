from typing import Dict, Union

MedRecordAttribute = Union[str, int]
MedRecordValue = Union[str, int, float, bool, None]
NodeIndex = MedRecordAttribute
EdgeIndex = int
Group = MedRecordAttribute
Attributes = Dict[MedRecordAttribute, MedRecordValue]

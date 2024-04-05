from typing import Dict, Union

MedRecordAttribute = Union[str, int]
MedRecordValue = Union[str, int, float, bool]
NodeIndex = MedRecordAttribute
EdgeIndex = int
Group = MedRecordAttribute
Attributes = Dict[MedRecordAttribute, MedRecordValue]

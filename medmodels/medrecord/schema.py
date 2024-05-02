from typing import Dict, Optional

from medmodels._medmodels import PyGroupSchema, PySchema
from medmodels.medrecord.datatype import DataType
from medmodels.medrecord.types import Group, MedRecordAttribute


class GroupSchema:
    _group_schema: PyGroupSchema

    def __init__(
        self,
        *,
        nodes: Dict[MedRecordAttribute, DataType] = {},
        edges: Dict[MedRecordAttribute, DataType] = {},
        strict: bool = False,
    ) -> None:
        self._group_schema = PyGroupSchema(
            nodes={x: nodes[x]._inner() for x in nodes},
            edges={x: edges[x]._inner() for x in edges},
            strict=strict,
        )


class Schema:
    _schema: PySchema

    def __init__(
        self,
        *,
        groups: Dict[Group, GroupSchema] = {},
        default: Optional[GroupSchema] = None,
        strict: bool = False,
    ) -> None:
        if default is not None:
            self._schema = PySchema(
                groups={x: groups[x]._group_schema for x in groups},
                default=default._group_schema,
                strict=strict,
            )
        else:
            self._schema = PySchema(
                groups={x: groups[x]._group_schema for x in groups},
                strict=strict,
            )

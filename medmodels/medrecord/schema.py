"""This module contains the schema classes for the medrecord module."""

from __future__ import annotations

from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

from medmodels._medmodels import (
    PyAttributeDataType,
    PyAttributeType,
    PyGroupSchema,
    PySchema,
)
from medmodels.medrecord.datatype import (
    DataType,
    DateTime,
    Duration,
    Float,
    Int,
    Option,
)
from medmodels.medrecord.datatype import (
    Union as DataTypeUnion,
)
from medmodels.medrecord.types import MedRecordAttribute

if TYPE_CHECKING:
    from medmodels.medrecord.types import Group


class AttributeType(Enum):
    """Enumeration of attribute types."""

    Categorical = auto()
    Continuous = auto()
    Temporal = auto()
    Unstructured = auto()

    @staticmethod
    def _from_py_attribute_type(py_attribute_type: PyAttributeType) -> AttributeType:
        """Converts a PyAttributeType to an AttributeType.

        Args:
            py_attribute_type (PyAttributeType): The PyAttributeType to convert.

        Returns:
            AttributeType: The converted AttributeType.
        """
        if py_attribute_type == PyAttributeType.Categorical:
            return AttributeType.Categorical
        if py_attribute_type == PyAttributeType.Continuous:
            return AttributeType.Continuous
        if py_attribute_type == PyAttributeType.Temporal:
            return AttributeType.Temporal
        if py_attribute_type == PyAttributeType.Unstructured:
            return AttributeType.Unstructured
        return None

    def _into_py_attribute_type(self) -> PyAttributeType:
        """Converts an AttributeType to a PyAttributeType.

        Returns:
            PyAttributeType: The converted PyAttributeType.
        """
        if self == AttributeType.Categorical:
            return PyAttributeType.Categorical
        if self == AttributeType.Continuous:
            return PyAttributeType.Continuous
        if self == AttributeType.Temporal:
            return PyAttributeType.Temporal
        msg = "Should never be reached"
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        """Returns a string representation of the AttributeType instance.

        Returns:
            str: String representation of the attribute type.
        """
        return f"AttributeType.{self.name}"

    def __str__(self) -> str:
        """Returns a string representation of the AttributeType instance.

        Returns:
            str: String representation of the attribute type.
        """
        return self.name

    def __eq__(self, value: object) -> bool:
        """Compares the AttributeType instance to another object for equality.

        Args:
            value (object): The object to compare against.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(value, PyAttributeType):
            return self._into_py_attribute_type() == value
        if isinstance(value, AttributeType):
            return str(self) == str(value)

        return False


CategoricalType: TypeAlias = DataType
CategoricalPair = Tuple[CategoricalType, Literal[AttributeType.Categorical]]

ContinuousType: TypeAlias = Union[
    Int, Float, Option[Int], Option[Float], "ContinuousUnionRecursive"
]
ContinuousUnionRecursive: TypeAlias = DataTypeUnion[ContinuousType, ContinuousType]
ContinuousPair = Tuple[ContinuousType, Literal[AttributeType.Continuous]]

TemporalType = Union[
    DateTime, Duration, Option[DateTime], Option[Duration], "TemporalUnionRecursive"
]
TemporalUnionRecursive: TypeAlias = DataTypeUnion[TemporalType, TemporalType]
TemporalPair = Tuple[TemporalType, Literal[AttributeType.Temporal]]

UnstructuredType: TypeAlias = DataType
UnstructuredPair = Tuple[UnstructuredType, Literal[AttributeType.Unstructured]]

AttributeDataType = Union[
    CategoricalPair, ContinuousPair, TemporalPair, UnstructuredPair
]

AttributesSchema = Dict[MedRecordAttribute, AttributeDataType]


class GroupSchema:
    """A schema for a group of nodes and edges."""

    _group_schema: PyGroupSchema

    def __init__(
        self,
        *,
        nodes: Optional[
            Dict[
                MedRecordAttribute,
                Union[DataType, AttributeDataType],
            ],
        ] = None,
        edges: Optional[
            Dict[
                MedRecordAttribute,
                Union[DataType, AttributeDataType],
            ],
        ] = None,
    ) -> None:
        """Initializes a new instance of GroupSchema.

        Args:
            nodes (Dict[MedRecordAttribute, Union[DataType, AttributeDataType]]):
                A dictionary mapping node attributes to their data
                types and optional attribute types. Defaults to an empty dictionary.
                When no attribute type is provided, it is inferred from the data type.
            edges (Dict[MedRecordAttribute, Union[DataType, AttributeDataType]]):
                A dictionary mapping edge attributes to their data types and
                optional attribute types. Defaults to an empty dictionary.
                When no attribute type is provided, it is inferred from the data type.
        """
        if edges is None:
            edges = {}
        if nodes is None:
            nodes = {}

        def _convert_input(
            input: Union[DataType, AttributeDataType],
        ) -> PyAttributeDataType:
            if isinstance(input, tuple):
                return PyAttributeDataType(
                    input[0]._inner(), input[1]._into_py_attribute_type()
                )

            return PyAttributeDataType(
                input._inner(), PyAttributeType.infer_from(input._inner())
            )

        self._group_schema = PyGroupSchema(
            nodes={x: _convert_input(nodes[x]) for x in nodes},
            edges={x: _convert_input(edges[x]) for x in edges},
        )

    @classmethod
    def _from_pygroupschema(cls, group_schema: PyGroupSchema) -> GroupSchema:
        """Creates a GroupSchema instance from an existing PyGroupSchema.

        Args:
            group_schema (PyGroupSchema): The PyGroupSchema instance to convert.

        Returns:
            GroupSchema: A new GroupSchema instance.
        """
        new_group_schema = cls()
        new_group_schema._group_schema = group_schema
        return new_group_schema

    @property
    def nodes(self) -> AttributesSchema:
        """Returns the node attributes in the GroupSchema instance.

        Returns:
            AttributesSchema: An AttributesSchema object containing the node attributes
                and their data types.
        """

        def _convert_node(
            input: PyAttributeDataType,
        ) -> AttributeDataType:
            # SAFETY: The typing is guaranteed to be correct
            return (
                DataType._from_py_data_type(input.data_type),
                AttributeType._from_py_attribute_type(input.attribute_type),
            )  # pyright: ignore[reportReturnType]

        return {
            x: _convert_node(self._group_schema.nodes[x])
            for x in self._group_schema.nodes
        }

    @property
    def edges(self) -> AttributesSchema:
        """Returns the edge attributes in the GroupSchema instance.

        Returns:
            AttributesSchema: An AttributesSchema object containing the edge attributes
                and their data types.
        """

        def _convert_edge(
            input: PyAttributeDataType,
        ) -> AttributeDataType:
            # SAFETY: The typing is guaranteed to be correct
            return (
                DataType._from_py_data_type(input.data_type),
                AttributeType._from_py_attribute_type(input.attribute_type),
            )  # pyright: ignore[reportReturnType]

        return {
            x: _convert_edge(self._group_schema.edges[x])
            for x in self._group_schema.edges
        }


class Schema:
    """A schema for a collection of groups."""

    _schema: PySchema

    def __init__(
        self,
        *,
        groups: Optional[Dict[Group, GroupSchema]] = None,
        default: Optional[GroupSchema] = None,
    ) -> None:
        """Initializes a new instance of Schema.

        Args:
            groups (Dict[Group, GroupSchema], optional): A dictionary of group names
                to their schemas. Defaults to an empty dictionary.
            default (Optional[GroupSchema], optional): The default group schema.
                If not provided, an empty group schema is used. Defaults to None.
        """
        if not default:
            default = GroupSchema()

        if groups is None:
            groups = {}

        self._schema = PySchema(
            groups={x: groups[x]._group_schema for x in groups},
            default=default._group_schema,
        )

    @classmethod
    def _from_py_schema(cls, schema: PySchema) -> Schema:
        """Creates a Schema instance from an existing PySchema.

        Args:
            schema (PySchema): The PySchema instance to convert.

        Returns:
            Schema: A new Schema instance.
        """
        new_schema = cls()
        new_schema._schema = schema
        return new_schema

    @property
    def groups(self) -> List[Group]:
        """Lists all the groups in the Schema instance.

        Returns:
            List[Group]: A list of groups.
        """
        return self._schema.groups

    def group(self, group: Group) -> GroupSchema:
        """Retrieves the schema for a specific group.

        Args:
            group (Group): The name of the group.

        Returns:
            GroupSchema: The schema for the specified group.

        Raises:
            ValueError: If the group does not exist in the schema.
        """  # noqa: DOC502
        return GroupSchema._from_pygroupschema(self._schema.group(group))

    @property
    def default(self) -> GroupSchema:
        """Retrieves the default group schema.

        Returns:
            Optional[GroupSchema]: The default group schema if it exists, otherwise
                None.
        """
        return GroupSchema._from_pygroupschema(self._schema.default)

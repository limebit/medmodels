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
    overload,
)

from medmodels._medmodels import (
    PyAttributeDataType,
    PyAttributeType,
    PyGroupSchema,
    PySchema,
    PySchemaType,
)
from medmodels.medrecord.datatype import (
    DataType,
    DateTime,
    Duration,
    Float,
    Int,
    Null,
    Option,
)
from medmodels.medrecord.datatype import Union as DataTypeUnion
from medmodels.medrecord.types import (
    Attributes,
    EdgeIndex,
    MedRecordAttribute,
    NodeIndex,
)

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
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
        msg = "Should never be reached"
        raise NotImplementedError(msg)

    @staticmethod
    def infer(data_type: DataType) -> AttributeType:
        """Infers the attribute type from the data type.

        Args:
            data_type (DataType): The data type to infer the attribute type from.

        Returns:
            AttributeType: The inferred attribute type.
        """
        return AttributeType._from_py_attribute_type(
            PyAttributeType.infer(data_type._inner())
        )

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
        if self == AttributeType.Unstructured:
            return PyAttributeType.Unstructured
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

    def __hash__(self) -> int:
        """Returns the hash of the AttributeType instance.

        Returns:
            int: The hash of the AttributeType instance.
        """
        return hash(self.name)

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
CategoricalPair: TypeAlias = Tuple[CategoricalType, Literal[AttributeType.Categorical]]

ContinuousType: TypeAlias = Union[
    Int,
    Float,
    Null,
    Option["ContinuousType"],
    DataTypeUnion["ContinuousType", "ContinuousType"],
]
ContinuousPair: TypeAlias = Tuple[ContinuousType, Literal[AttributeType.Continuous]]

TemporalType = Union[
    DateTime,
    Duration,
    Null,
    Option["TemporalType"],
    DataTypeUnion["TemporalType", "TemporalType"],
]
TemporalPair: TypeAlias = Tuple[TemporalType, Literal[AttributeType.Temporal]]

UnstructuredType: TypeAlias = DataType
UnstructuredPair: TypeAlias = Tuple[
    UnstructuredType, Literal[AttributeType.Unstructured]
]

AttributeDataType: TypeAlias = Union[
    CategoricalPair, ContinuousPair, TemporalPair, UnstructuredPair
]

AttributesSchema: TypeAlias = Dict[MedRecordAttribute, AttributeDataType]


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
                input._inner(), PyAttributeType.infer(input._inner())
            )

        self._group_schema = PyGroupSchema(
            nodes={x: _convert_input(nodes[x]) for x in nodes},
            edges={x: _convert_input(edges[x]) for x in edges},
        )

    @classmethod
    def _from_py_group_schema(cls, group_schema: PyGroupSchema) -> GroupSchema:
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

    def validate_node(self, index: NodeIndex, attributes: Attributes) -> None:
        """Validates the attributes of a node against the schema.

        Args:
            index (NodeIndex): The index of the node.
            attributes (Attributes): The attributes of the node.
        """
        self._group_schema.validate_node(index, attributes)

    def validate_edge(self, index: EdgeIndex, attributes: Attributes) -> None:
        """Validates the attributes of an edge against the schema.

        Args:
            index (EdgeIndex): The index of the edge.
            attributes (Attributes): The attributes of the edge.
        """
        self._group_schema.validate_edge(index, attributes)


class SchemaType(Enum):
    """Enumeration of schema types."""

    Provided = auto()
    Inferred = auto()

    @staticmethod
    def _from_py_schema_type(py_schema_type: PySchemaType) -> SchemaType:
        """Converts a PySchemaType to a SchemaType.

        Args:
            py_schema_type (PySchemaType): The PySchemaType to convert.

        Returns:
            SchemaType: The converted SchemaType.
        """
        if py_schema_type == PySchemaType.Provided:
            return SchemaType.Provided
        if py_schema_type == PySchemaType.Inferred:
            return SchemaType.Inferred

        msg = "Should never be reached"
        raise NotImplementedError(msg)

    def _into_py_schema_type(self) -> PySchemaType:
        """Converts a SchemaType to a PySchemaType.

        Returns:
            PySchemaType: The converted PySchemaType.
        """
        if self == SchemaType.Provided:
            return PySchemaType.Provided
        if self == SchemaType.Inferred:
            return PySchemaType.Inferred

        msg = "Should never be reached"
        raise NotImplementedError(msg)


class Schema:
    """A schema for a collection of groups."""

    _schema: PySchema

    def __init__(
        self,
        *,
        groups: Optional[Dict[Group, GroupSchema]] = None,
        ungrouped: Optional[GroupSchema] = None,
        schema_type: Optional[SchemaType] = None,
    ) -> None:
        """Initializes a new instance of Schema.

        Args:
            groups (Dict[Group, GroupSchema], optional): A dictionary of group names
                to their schemas. Defaults to None.
            ungrouped (Optional[GroupSchema], optional): The group schema for all nodes
                not in a group. If not provided, an empty group schema is used.
                Defaults to None.
            schema_type (Optional[SchemaType], optional): The type of the schema.
                If not provided, the schema is of type provided. Defaults to None.
        """
        if not ungrouped:
            ungrouped = GroupSchema()

        if groups is None:
            groups = {}

        if schema_type:
            self._schema = PySchema(
                groups={x: groups[x]._group_schema for x in groups},
                ungrouped=ungrouped._group_schema,
                schema_type=schema_type._into_py_schema_type(),
            )
        else:
            self._schema = PySchema(
                groups={x: groups[x]._group_schema for x in groups},
                ungrouped=ungrouped._group_schema,
            )

    @classmethod
    def infer(cls, medrecord: MedRecord) -> Schema:
        """Infers a schema from a MedRecord instance.

        Args:
            medrecord (MedRecord): The MedRecord instance to infer the schema from.

        Returns:
            Schema: The inferred schema.
        """
        new_schema = cls()
        new_schema._schema = PySchema.infer(medrecord._medrecord)
        return new_schema

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
        return GroupSchema._from_py_group_schema(self._schema.group(group))

    @property
    def ungrouped(self) -> GroupSchema:
        """Retrieves the group schema for all ungrouped nodes and edges.

        Returns:
            GroupSchema: The ungrouped group schema.
        """
        return GroupSchema._from_py_group_schema(self._schema.ungrouped)

    @property
    def schema_type(self) -> SchemaType:
        """Retrieves the schema type.

        Returns:
            SchemaType: The schema type.
        """
        return SchemaType._from_py_schema_type(self._schema.schema_type)

    def validate_node(
        self, index: NodeIndex, attributes: Attributes, group: Optional[Group] = None
    ) -> None:
        """Validates the attributes of a node against the schema.

        Args:
            index (NodeIndex): The index of the node.
            attributes (Attributes): The attributes of the node.
            group (Optional[Group], optional): The group to validate the node against.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        self._schema.validate_node(index, attributes, group)

    def validate_edge(
        self, index: EdgeIndex, attributes: Attributes, group: Optional[Group] = None
    ) -> None:
        """Validates the attributes of an edge against the schema.

        Args:
            index (EdgeIndex): The index of the edge.
            attributes (Attributes): The attributes of the edge.
            group (Optional[Group], optional): The group to validate the edge against.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        self._schema.validate_edge(index, attributes, group)

    @overload
    def set_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[
            Literal[AttributeType.Categorical, AttributeType.Unstructured]
        ] = None,
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def set_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: ContinuousType,
        attribute_type: Literal[AttributeType.Continuous],
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def set_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: TemporalType,
        attribute_type: Literal[AttributeType.Temporal],
        group: Optional[Group] = None,
    ) -> None: ...

    def set_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[AttributeType] = None,
        group: Optional[Group] = None,
    ) -> None:
        """Sets the data type and attribute type of a node attribute.

        If a data type for the attribute already exists, it is overwritten.

        Args:
            attribute (MedRecordAttribute): The name of the attribute.
            data_type (DataType): The data type of the attribute.
            attribute_type (Optional[AttributeType], optional): The attribute type of
                the attribute. If not provided, the attribute type is inferred
                from the data type. Defaults to None.
            group (Optional[Group], optional): The group to set the attribute for.
                If no schema for the group exists, a new schema is created.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        if not attribute_type:
            attribute_type = AttributeType.infer(data_type)

        self._schema.set_node_attribute(
            attribute,
            data_type._inner(),
            attribute_type._into_py_attribute_type(),
            group,
        )

    @overload
    def set_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[
            Literal[AttributeType.Categorical, AttributeType.Unstructured]
        ] = None,
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def set_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: ContinuousType,
        attribute_type: Literal[AttributeType.Continuous],
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def set_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: TemporalType,
        attribute_type: Literal[AttributeType.Temporal],
        group: Optional[Group] = None,
    ) -> None: ...

    def set_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[AttributeType] = None,
        group: Optional[Group] = None,
    ) -> None:
        """Sets the data type and attribute type of an edge attribute.

        If a data type for the attribute already exists, it is overwritten.

        Args:
            attribute (MedRecordAttribute): The name of the attribute.
            data_type (DataType): The data type of the attribute.
            attribute_type (Optional[AttributeType], optional): The attribute type of
                the attribute. If not provided, the attribute type is inferred
                from the data type. Defaults to None.
            group (Optional[Group], optional): The group to set the attribute for.
                If no schema for this group exists, a new schema is created.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        if not attribute_type:
            attribute_type = AttributeType.infer(data_type)

        self._schema.set_edge_attribute(
            attribute,
            data_type._inner(),
            attribute_type._into_py_attribute_type(),
            group,
        )

    @overload
    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[
            Literal[AttributeType.Categorical, AttributeType.Unstructured]
        ] = None,
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: ContinuousType,
        attribute_type: Literal[AttributeType.Continuous],
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: TemporalType,
        attribute_type: Literal[AttributeType.Temporal],
        group: Optional[Group] = None,
    ) -> None: ...

    def update_node_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[AttributeType] = None,
        group: Optional[Group] = None,
    ) -> None:
        """Updates the data type and attribute type of a node attribute.

        If a data type for the attribute already exists, it is merged
        with the new data type.

        Args:
            attribute (MedRecordAttribute): The name of the attribute.
            data_type (DataType): The data type of the attribute.
            attribute_type (Optional[AttributeType], optional): The attribute type of
                the attribute. If not provided, the attribute type is inferred
                from the data type. Defaults to None.
            group (Optional[Group], optional): The group to update the attribute for.
                If no schema for this group exists, a new schema is created.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        if not attribute_type:
            attribute_type = AttributeType.infer(data_type)

        self._schema.update_node_attribute(
            attribute,
            data_type._inner(),
            attribute_type._into_py_attribute_type(),
            group,
        )

    @overload
    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[
            Literal[AttributeType.Categorical, AttributeType.Unstructured]
        ] = None,
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: ContinuousType,
        attribute_type: Literal[AttributeType.Continuous],
        group: Optional[Group] = None,
    ) -> None: ...

    @overload
    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: TemporalType,
        attribute_type: Literal[AttributeType.Temporal],
        group: Optional[Group] = None,
    ) -> None: ...

    def update_edge_attribute(
        self,
        attribute: MedRecordAttribute,
        data_type: DataType,
        attribute_type: Optional[AttributeType] = None,
        group: Optional[Group] = None,
    ) -> None:
        """Updates the data type and attribute type of an edge attribute.

        If a data type for the attribute already exists, it is merged
        with the new data type.

        Args:
            attribute (MedRecordAttribute): The name of the attribute.
            data_type (DataType): The data type of the attribute.
            attribute_type (Optional[AttributeType], optional): The attribute type of
                the attribute. If not provided, the attribute type is inferred
                from the data type. Defaults to None.
            group (Optional[Group], optional): The group to update the attribute for.
                If no schema for this group exists, a new schema is created.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        if not attribute_type:
            attribute_type = AttributeType.infer(data_type)

        self._schema.update_edge_attribute(
            attribute,
            data_type._inner(),
            attribute_type._into_py_attribute_type(),
            group,
        )

    def remove_node_attribute(
        self, attribute: MedRecordAttribute, group: Optional[Group] = None
    ) -> None:
        """Removes a node attribute from the schema.

        Args:
            attribute (MedRecordAttribute): The name of the attribute to remove.
            group (Optional[Group], optional): The group to remove the attribute from.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        self._schema.remove_node_attribute(attribute, group)

    def remove_edge_attribute(
        self, attribute: MedRecordAttribute, group: Optional[Group] = None
    ) -> None:
        """Removes an edge attribute from the schema.

        Args:
            attribute (MedRecordAttribute): The name of the attribute to remove.
            group (Optional[Group], optional): The group to remove the attribute from.
                If not provided, the ungrouped schema is used. Defaults to None.
        """
        self._schema.remove_edge_attribute(attribute, group)

    def add_group(self, group: Group, group_schema: GroupSchema) -> None:
        """Adds a new group to the schema.

        Args:
            group (Group): The name of the group.
            group_schema (GroupSchema): The schema for the group.
        """
        self._schema.add_group(group, group_schema._group_schema)

    def remove_group(self, group: Group) -> None:
        """Removes a group from the schema.

        Args:
            group (Group): The name of the group to remove.
        """
        self._schema.remove_group(group)

    def freeze(self) -> None:
        """Freezes the schema. No changes are automatically inferred."""
        self._schema.freeze()

    def unfreeze(self) -> None:
        """Unfreezes the schema. Changes are automatically inferred."""
        self._schema.unfreeze()

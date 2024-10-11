from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, overload

from medmodels._medmodels import (
    PyAttributeDataType,
    PyAttributeType,
    PyGroupSchema,
    PySchema,
)
from medmodels.medrecord.datatype import DataType
from medmodels.medrecord.types import Group, MedRecordAttribute


class AttributeType(Enum):
    Categorical = auto()
    Continuous = auto()
    Temporal = auto()

    @staticmethod
    def _from_py_attribute_type(py_attribute_type: PyAttributeType) -> AttributeType:
        """
        Converts a PyAttributeType to an AttributeType.

        Args:
            py_attribute_type (PyAttributeType): The PyAttributeType to convert.

        Returns:
            AttributeType: The converted AttributeType.
        """
        if py_attribute_type == PyAttributeType.Categorical:
            return AttributeType.Categorical
        elif py_attribute_type == PyAttributeType.Continuous:
            return AttributeType.Continuous
        elif py_attribute_type == PyAttributeType.Temporal:
            return AttributeType.Temporal

    def _into_py_attribute_type(self) -> PyAttributeType:
        """
        Converts an AttributeType to a PyAttributeType.

        Returns:
            PyAttributeType: The converted PyAttributeType.
        """
        if self == AttributeType.Categorical:
            return PyAttributeType.Categorical
        elif self == AttributeType.Continuous:
            return PyAttributeType.Continuous
        elif self == AttributeType.Temporal:
            return PyAttributeType.Temporal
        else:
            raise NotImplementedError("Should never be reached")

    def __repr__(self) -> str:
        """
        Returns a string representation of the AttributeType instance.

        Returns:
            str: String representation of the attribute type.
        """
        return f"AttributeType.{self.name}"

    def __str__(self) -> str:
        """
        Returns a string representation of the AttributeType instance.

        Returns:
            str: String representation of the attribute type.
        """
        return self.name

    def __eq__(self, value: object) -> bool:
        """
        Compares the AttributeType instance to another object for equality.

        Args:
            value (object): The object to compare against.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(value, PyAttributeType):
            return self._into_py_attribute_type() == value
        elif isinstance(value, AttributeType):
            return str(self) == str(value)

        return False


class AttributesSchema:
    _attributes_schema: Dict[
        MedRecordAttribute, Tuple[DataType, Optional[AttributeType]]
    ]

    def __init__(
        self,
        attributes_schema: Dict[
            MedRecordAttribute, Tuple[DataType, Optional[AttributeType]]
        ],
    ) -> None:
        """
        Initializes a new instance of AttributesSchema.

        Args:
            attributes_schema (Dict[MedRecordAttribute, Tuple[DataType, Optional[AttributeType]]]):
                A dictionary mapping MedRecordAttributes to their data types and
                optional attribute types.

        Returns:
            None
        """
        self._attributes_schema = attributes_schema

    def __repr__(self) -> str:
        """
        Returns a string representation of the AttributesSchema instance.

        Returns:
            str: String representation of the attribute schema.
        """
        return self._attributes_schema.__repr__()

    def __getitem__(
        self, key: MedRecordAttribute
    ) -> Tuple[DataType, Optional[AttributeType]]:
        """
        Gets the data type and optional attribute type for a given MedRecordAttribute.

        Args:
            key (MedRecordAttribute): The attribute for which the data type is
                requested.

        Returns:
            Tuple[DataType, Optional[AttributeType]]: The data type and optional
                attribute type of the given attribute.
        """
        return self._attributes_schema[key]

    def __contains__(self, key: MedRecordAttribute) -> bool:
        """
        Checks if a given MedRecordAttribute is in the attributes schema.

        Args:
            key (MedRecordAttribute): The attribute to check.

        Returns:
            bool: True if the attribute exists in the schema, False otherwise.
        """
        return key in self._attributes_schema

    def __iter__(self):
        """
        Returns an iterator over the attributes schema.

        Returns:
            Iterator: An iterator over the attribute keys.
        """
        return self._attributes_schema.__iter__()

    def __len__(self) -> int:
        """
        Returns the number of attributes in the schema.

        Returns:
            int: The number of attributes.
        """
        return len(self._attributes_schema)

    def __eq__(self, value: object) -> bool:
        """
        Compares the AttributesSchema instance to another object for equality.

        Args:
            value (object): The object to compare against.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not (isinstance(value, AttributesSchema) or isinstance(value, dict)):
            return False

        attribute_schema = (
            value._attributes_schema if isinstance(value, AttributesSchema) else value
        )

        if not attribute_schema.keys() == self._attributes_schema.keys():
            return False

        for key in self._attributes_schema.keys():
            if (
                not isinstance(attribute_schema[key], tuple)
                or not isinstance(
                    attribute_schema[key][0], type(self._attributes_schema[key][0])
                )
                or attribute_schema[key][1] != self._attributes_schema[key][1]
            ):
                return False

        return True

    def keys(self):
        """
        Returns the attribute keys in the schema.

        Returns:
            KeysView: A view object displaying a list of dictionary's keys.
        """
        return self._attributes_schema.keys()

    def values(self):
        """
        Returns the attribute values in the schema.

        Returns:
            ValuesView: A view object displaying a list of dictionary's values.
        """
        return self._attributes_schema.values()

    def items(self):
        """
        Returns the attribute key-value pairs in the schema.

        Returns:
            ItemsView: A set-like object providing a view on D's items.
        """
        return self._attributes_schema.items()

    @overload
    def get(
        self, key: MedRecordAttribute
    ) -> Optional[Tuple[DataType, Optional[AttributeType]]]: ...

    @overload
    def get(
        self, key: MedRecordAttribute, default: Tuple[DataType, Optional[AttributeType]]
    ) -> Tuple[DataType, Optional[AttributeType]]: ...

    def get(
        self,
        key: MedRecordAttribute,
        default: Optional[Tuple[DataType, Optional[AttributeType]]] = None,
    ) -> Optional[Tuple[DataType, Optional[AttributeType]]]:
        """
        Gets the data type and optional attribute type for a given attribute, returning
        a default value if the attribute is not present.

        Args:
            key (MedRecordAttribute): The attribute for which the data type is
                requested.
            default (Optional[Tuple[DataType, Optional[AttributeType]]], optional):
                The default data type and attribute type to return if the attribute
                is not found. Defaults to None.

        Returns:
            Optional[Tuple[DataType, Optional[AttributeType]]]: The data type and
                optional attribute type of the given attribute or the default value.
        """
        return self._attributes_schema.get(key, default)


class GroupSchema:
    _group_schema: PyGroupSchema

    def __init__(
        self,
        *,
        nodes: Dict[
            MedRecordAttribute, Union[DataType, Tuple[DataType, AttributeType]]
        ] = {},
        edges: Dict[
            MedRecordAttribute, Union[DataType, Tuple[DataType, AttributeType]]
        ] = {},
        strict: bool = False,
    ) -> None:
        """
        Initializes a new instance of GroupSchema.

        Args:
            nodes (Dict[MedRecordAttribute, Union[DataType, Tuple[DataType, AttributeType]]]):
                A dictionary mapping node attributes to their data
                types and optional attribute types. Defaults to an empty dictionary.
            edges (Dict[MedRecordAttribute, Union[DataType, Tuple[DataType, AttributeType]]]):
                A dictionary mapping edge attributes to their data types and
                optional attribute types. Defaults to an empty dictionary.
            strict (bool, optional): Indicates whether the schema should be strict.
                Defaults to False.

        Returns:
            None
        """

        def _convert_input(
            input: Union[DataType, Tuple[DataType, AttributeType]],
        ) -> PyAttributeDataType:
            if isinstance(input, tuple):
                return PyAttributeDataType(
                    input[0]._inner(), input[1]._into_py_attribute_type()
                )
            return PyAttributeDataType(input._inner(), None)

        self._group_schema = PyGroupSchema(
            nodes={x: _convert_input(nodes[x]) for x in nodes},
            edges={x: _convert_input(edges[x]) for x in edges},
            strict=strict,
        )

    @classmethod
    def _from_pygroupschema(cls, group_schema: PyGroupSchema) -> GroupSchema:
        """
        Creates a GroupSchema instance from an existing PyGroupSchema.

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
        """
        Returns the node attributes in the GroupSchema instance.

        Returns:
            AttributesSchema: An AttributesSchema object containing the node attributes
                and their data types.
        """

        def _convert_node(
            input: PyAttributeDataType,
        ) -> Tuple[DataType, Optional[AttributeType]]:
            return (
                DataType._from_py_data_type(input.data_type),
                AttributeType._from_py_attribute_type(input.attribute_type)
                if input.attribute_type is not None
                else None,
            )

        return AttributesSchema(
            {
                x: _convert_node(self._group_schema.nodes[x])
                for x in self._group_schema.nodes
            }
        )

    @property
    def edges(self) -> AttributesSchema:
        """
        Returns the edge attributes in the GroupSchema instance.

        Returns:
            AttributesSchema: An AttributesSchema object containing the edge attributes
                and their data types.
        """

        def _convert_edge(
            input: PyAttributeDataType,
        ) -> Tuple[DataType, Optional[AttributeType]]:
            return (
                DataType._from_py_data_type(input.data_type),
                AttributeType._from_py_attribute_type(input.attribute_type)
                if input.attribute_type is not None
                else None,
            )

        return AttributesSchema(
            {
                x: _convert_edge(self._group_schema.edges[x])
                for x in self._group_schema.edges
            }
        )

    @property
    def strict(self) -> Optional[bool]:
        """
        Indicates whether the GroupSchema instance is strict.

        Returns:
            Optional[bool]: True if the schema is strict, False otherwise.
        """
        return self._group_schema.strict


class Schema:
    _schema: PySchema

    def __init__(
        self,
        *,
        groups: Dict[Group, GroupSchema] = {},
        default: Optional[GroupSchema] = None,
        strict: bool = False,
    ) -> None:
        """
        Initializes a new instance of Schema.

        Args:
            groups (Dict[Group, GroupSchema], optional): A dictionary of group names
                to their schemas. Defaults to an empty dictionary.
            default (Optional[GroupSchema], optional): The default group schema.
                Defaults to None.
            strict (bool, optional): Indicates whether the schema should be strict.
                Defaults to False.

        Returns:
            None
        """
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

    @classmethod
    def _from_py_schema(cls, schema: PySchema) -> Schema:
        """
        Creates a Schema instance from an existing PySchema.

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
        """
        Lists all the groups in the Schema instance.

        Returns:
            List[Group]: A list of groups.
        """
        return self._schema.groups

    def group(self, group: Group) -> GroupSchema:
        """
        Retrieves the schema for a specific group.

        Args:
            group (Group): The name of the group.

        Returns:
            GroupSchema: The schema for the specified group.

        Raises:
            ValueError: If the specified group does not exist.
        """
        return GroupSchema._from_pygroupschema(self._schema.group(group))

    @property
    def default(self) -> Optional[GroupSchema]:
        """
        Retrieves the default group schema.

        Returns:
            Optional[GroupSchema]: The default group schema if it exists, otherwise
                None.
        """
        if self._schema.default is None:
            return None

        return GroupSchema._from_pygroupschema(self._schema.default)

    @property
    def strict(self) -> Optional[bool]:
        """
        Indicates whether the Schema instance is strict.

        Returns:
            Optional[bool]: True if the schema is strict, False otherwise.
        """
        return self._schema.strict

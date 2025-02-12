"""MedRecord-associated data types."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

from medmodels._medmodels import (
    PyAny,
    PyBool,
    PyDateTime,
    PyDuration,
    PyFloat,
    PyInt,
    PyNull,
    PyOption,
    PyString,
    PyUnion,
)

PyDataType: TypeAlias = typing.Union[
    PyString,
    PyInt,
    PyFloat,
    PyBool,
    PyDateTime,
    PyDuration,
    PyNull,
    PyAny,
    PyUnion,
    PyOption,
]


class DataType(ABC):
    """Abstract class for data types."""

    @abstractmethod
    def _inner(self) -> PyDataType: ...

    @abstractmethod
    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        ...

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Checks if the data type is equal to another data type."""
        ...

    @staticmethod
    def _from_py_data_type(datatype: PyDataType) -> DataType:
        if isinstance(datatype, PyString):
            return String()
        if isinstance(datatype, PyInt):
            return Int()
        if isinstance(datatype, PyFloat):
            return Float()
        if isinstance(datatype, PyBool):
            return Bool()
        if isinstance(datatype, PyDateTime):
            return DateTime()
        if isinstance(datatype, PyDuration):
            return Duration()
        if isinstance(datatype, PyNull):
            return Null()
        if isinstance(datatype, PyAny):
            return Any()
        if isinstance(datatype, PyUnion):
            return Union(
                DataType._from_py_data_type(datatype.dtype1),
                DataType._from_py_data_type(datatype.dtype2),
            )
        return Option(DataType._from_py_data_type(datatype.dtype))


class String(DataType):
    """Data type for strings."""

    _string: PyString

    def __init__(self) -> None:
        """Initializes the String data type."""
        self._string = PyString()

    def _inner(self) -> PyDataType:
        return self._string

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "String"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.String"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, String)


class Int(DataType):
    """Data type for integers."""

    _int: PyInt

    def __init__(self) -> None:
        """Initializes the Int data type."""
        self._int = PyInt()

    def _inner(self) -> PyDataType:
        return self._int

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Int"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Int"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Int)


class Float(DataType):
    """Data type for floating-point numbers."""

    _float: PyFloat

    def __init__(self) -> None:
        """Initializes the Float data type."""
        self._float = PyFloat()

    def _inner(self) -> PyDataType:
        return self._float

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Float"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Float"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Float)


class Bool(DataType):
    """Data type for boolean values."""

    _bool: PyBool

    def __init__(self) -> None:
        """Initializes the Bool data type."""
        self._bool = PyBool()

    def _inner(self) -> PyDataType:
        return self._bool

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Bool"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Bool"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Bool)


class DateTime(DataType):
    """Data type for date and time values."""

    _datetime: PyDateTime

    def __init__(self) -> None:
        """Initializes the DateTime data type."""
        self._datetime = PyDateTime()

    def _inner(self) -> PyDataType:
        return self._datetime

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "DateTime"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.DateTime"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, DateTime)


class Duration(DataType):
    """Data type for duration (timedelta)."""

    _duration: PyDuration

    def __init__(self) -> None:
        """Initializes the Duration data type."""
        self._duration = PyDuration()

    def _inner(self) -> PyDataType:
        return self._duration

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Duration"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Duration"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Duration)


class Null(DataType):
    """Data type for null values."""

    _null: PyNull

    def __init__(self) -> None:
        """Initializes the Null data type."""
        self._null = PyNull()

    def _inner(self) -> PyDataType:
        return self._null

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Null"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Null"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Null)


class Any(DataType):
    """Data type for any values."""

    _any: PyAny

    def __init__(self) -> None:
        """Initializes the Any data type."""
        self._any = PyAny()

    def _inner(self) -> PyDataType:
        return self._any

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return "Any"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return "DataType.Any"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Any)


U1 = TypeVar("U1", bound=DataType)
U2 = TypeVar("U2", bound=DataType)


class Union(DataType, Generic[U1, U2]):
    """Data type for unions of data types."""

    _union: PyUnion

    def __init__(self, dtype1: U1, dtype2: U2) -> None:
        """Initializes the Union data type.

        Args:
            dtype1 (U1): The first data type of the union.
            dtype2 (U2): The second data type of the union.
        """
        self._union = PyUnion(dtype1._inner(), dtype2._inner())

    def _inner(self) -> PyDataType:
        return self._union

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return f"Union({DataType._from_py_data_type(self._union.dtype1).__str__()}, {DataType._from_py_data_type(self._union.dtype2).__str__()})"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return f"DataType.Union({DataType._from_py_data_type(self._union.dtype1).__repr__()}, {DataType._from_py_data_type(self._union.dtype2).__repr__()})"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return (
            isinstance(value, Union)
            and DataType._from_py_data_type(self._union.dtype1)
            == DataType._from_py_data_type(value._union.dtype1)
            and DataType._from_py_data_type(self._union.dtype2)
            == DataType._from_py_data_type(value._union.dtype2)
        )


T = TypeVar("T", bound=DataType)


class Option(DataType, Generic[T]):
    """Data type for optional values."""

    _option: PyOption

    def __init__(self, dtype: T) -> None:
        """Initializes the Option data type.

        Args:
            dtype (T): The data type of the optional value.
        """
        self._option = PyOption(dtype._inner())

    def _inner(self) -> PyDataType:
        return self._option

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the data type."""
        return f"Option({DataType._from_py_data_type(self._option.dtype).__str__()})"

    def __repr__(self) -> str:
        """Returns an official string representation of the data type."""
        return f"DataType.Option({DataType._from_py_data_type(self._option.dtype).__repr__()})"

    def __eq__(self, value: object) -> bool:
        """Checks if the data type of the value is equal to this data type.

        Args:
            value (object): The value to compare.

        Returns:
            bool: True if the data type is equal to this data type, otherwise
                False.
        """
        return isinstance(value, Option) and DataType._from_py_data_type(
            self._option.dtype
        ) == DataType._from_py_data_type(value._option.dtype)

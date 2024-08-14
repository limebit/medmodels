from __future__ import annotations

import typing
from abc import ABCMeta, abstractmethod

from medmodels._medmodels import (
    PyAny,
    PyBool,
    PyDateTime,
    PyFloat,
    PyInt,
    PyNull,
    PyOption,
    PyString,
    PyUnion,
)

if typing.TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

PyDataType: TypeAlias = typing.Union[
    PyString,
    PyInt,
    PyFloat,
    PyBool,
    PyDateTime,
    PyNull,
    PyAny,
    PyUnion,
    PyOption,
]


class DataType(metaclass=ABCMeta):
    @abstractmethod
    def _inner(self) -> PyDataType: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __eq__(self, value: object) -> bool: ...

    @staticmethod
    def _from_py_data_type(datatype: PyDataType) -> DataType:
        if isinstance(datatype, PyString):
            return String()
        elif isinstance(datatype, PyInt):
            return Int()
        elif isinstance(datatype, PyFloat):
            return Float()
        elif isinstance(datatype, PyBool):
            return Bool()
        elif isinstance(datatype, PyDateTime):
            return DateTime()
        elif isinstance(datatype, PyNull):
            return Null()
        elif isinstance(datatype, PyAny):
            return Any()
        elif isinstance(datatype, PyUnion):
            return Union(
                DataType._from_py_data_type(datatype.dtype1),
                DataType._from_py_data_type(datatype.dtype2),
            )
        else:
            return Option(DataType._from_py_data_type(datatype.dtype))


class String(DataType):
    _string: PyString

    def __init__(self) -> None:
        self._string = PyString()

    def _inner(self) -> PyDataType:
        return self._string

    def __str__(self) -> str:
        return "String"

    def __repr__(self) -> str:
        return "DataType.String"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, String)


class Int(DataType):
    _int: PyInt

    def __init__(self) -> None:
        self._int = PyInt()

    def _inner(self) -> PyDataType:
        return self._int

    def __str__(self) -> str:
        return "Int"

    def __repr__(self) -> str:
        return "DataType.Int"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Int)


class Float(DataType):
    _float: PyFloat

    def __init__(self) -> None:
        self._float = PyFloat()

    def _inner(self) -> PyDataType:
        return self._float

    def __str__(self) -> str:
        return "Float"

    def __repr__(self) -> str:
        return "DataType.Float"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Float)


class Bool(DataType):
    _bool: PyBool

    def __init__(self) -> None:
        self._bool = PyBool()

    def _inner(self) -> PyDataType:
        return self._bool

    def __str__(self) -> str:
        return "Bool"

    def __repr__(self) -> str:
        return "DataType.Bool"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Bool)


class DateTime(DataType):
    _datetime: PyDateTime

    def __init__(self) -> None:
        self._datetime = PyDateTime()

    def _inner(self) -> PyDataType:
        return self._datetime

    def __str__(self) -> str:
        return "DateTime"

    def __repr__(self) -> str:
        return "DataType.DateTime"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DateTime)


class Null(DataType):
    _null: PyNull

    def __init__(self) -> None:
        self._null = PyNull()

    def _inner(self) -> PyDataType:
        return self._null

    def __str__(self) -> str:
        return "Null"

    def __repr__(self) -> str:
        return "DataType.Null"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Null)


class Any(DataType):
    _any: PyAny

    def __init__(self) -> None:
        self._any = PyAny()

    def _inner(self) -> PyDataType:
        return self._any

    def __str__(self) -> str:
        return "Any"

    def __repr__(self) -> str:
        return "DataType.Any"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Any)


class Union(DataType):
    _union: PyUnion

    def __init__(self, *dtypes: DataType) -> None:
        if len(dtypes) < 2:
            raise ValueError("Union must have at least two arguments")
        elif len(dtypes) == 2:
            self._union = PyUnion(dtypes[0]._inner(), dtypes[1]._inner())
        else:
            self._union = PyUnion(dtypes[0]._inner(), Union(*dtypes[1:])._inner())

    def _inner(self) -> PyDataType:
        return self._union

    def __str__(self) -> str:
        return f"Union({DataType._from_py_data_type(self._union.dtype1).__str__()}, {DataType._from_py_data_type(self._union.dtype2).__str__()})"

    def __repr__(self) -> str:
        return f"DataType.Union({DataType._from_py_data_type(self._union.dtype1).__repr__()}, {DataType._from_py_data_type(self._union.dtype2).__repr__()})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Union)
            and DataType._from_py_data_type(self._union.dtype1)
            == DataType._from_py_data_type(value._union.dtype1)
            and DataType._from_py_data_type(self._union.dtype2)
            == DataType._from_py_data_type(value._union.dtype2)
        )


class Option(DataType):
    _option: PyOption

    def __init__(self, dtype: DataType) -> None:
        self._option = PyOption(dtype._inner())

    def _inner(self) -> PyDataType:
        return self._option

    def __str__(self) -> str:
        return f"Option({DataType._from_py_data_type(self._option.dtype).__str__()})"

    def __repr__(self) -> str:
        return f"DataType.Option({DataType._from_py_data_type(self._option.dtype).__repr__()})"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Option) and DataType._from_py_data_type(
            self._option.dtype
        ) == DataType._from_py_data_type(value._option.dtype)

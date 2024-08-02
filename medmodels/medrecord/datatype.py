from __future__ import annotations

import typing
from abc import ABCMeta, abstractmethod

from medmodels._medmodels import (
    PyAny,
    PyBool,
    PyDateTime,
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


class String(DataType):
    _string: PyString

    def __init__(self) -> None:
        self._string = PyString()

    def _inner(self) -> PyDataType:
        return self._string


class Int(DataType):
    _int: PyInt

    def __init__(self) -> None:
        self._int = PyInt()

    def _inner(self) -> PyDataType:
        return self._int


class Bool(DataType):
    _bool: PyBool

    def __init__(self) -> None:
        self._bool = PyBool()

    def _inner(self) -> PyDataType:
        return self._bool


class DateTime(DataType):
    _datetime: PyDateTime

    def __init__(self) -> None:
        self._datetime = PyDateTime()

    def _inner(self) -> PyDataType:
        return self._datetime


class Null(DataType):
    _null: PyNull

    def __init__(self) -> None:
        self._null = PyNull()

    def _inner(self) -> PyDataType:
        return self._null


class Any(DataType):
    _any: PyAny

    def __init__(self) -> None:
        self._any = PyAny()

    def _inner(self) -> PyDataType:
        return self._any


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


class Option(DataType):
    _option: PyOption

    def __init__(self, dtype: DataType) -> None:
        self._option = PyOption(dtype._inner())

    def _inner(self) -> PyDataType:
        return self._option

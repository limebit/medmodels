import unittest

import medmodels.medrecord as mr
from medmodels._medmodels import (
    PyAny,
    PyBool,
    PyInt,
    PyNull,
    PyOption,
    PyString,
    PyUnion,
)


class TestDataType(unittest.TestCase):
    def test_string(self):
        string = mr.String()
        self.assertTrue(isinstance(string._inner(), PyString))

    def test_int(self):
        integer = mr.Int()
        self.assertTrue(isinstance(integer._inner(), PyInt))

    def test_bool(self):
        bool = mr.Bool()
        self.assertTrue(isinstance(bool._inner(), PyBool))

    def test_null(self):
        null = mr.Null()
        self.assertTrue(isinstance(null._inner(), PyNull))

    def test_any(self):
        any = mr.Any()
        self.assertTrue(isinstance(any._inner(), PyAny))

    def test_union(self):
        union = mr.Union(mr.String(), mr.Int())
        self.assertTrue(isinstance(union._inner(), PyUnion))

        union = mr.Union(mr.String(), mr.Int(), mr.Bool())
        self.assertTrue(isinstance(union._inner(), PyUnion))

    def test_invalid_union(self):
        with self.assertRaises(ValueError):
            mr.Union(mr.String())

    def test_option(self):
        option = mr.Option(mr.String())
        self.assertTrue(isinstance(option._inner(), PyOption))

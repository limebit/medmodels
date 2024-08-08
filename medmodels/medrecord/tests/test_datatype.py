import unittest

import medmodels.medrecord as mr
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


class TestDataType(unittest.TestCase):
    def test_string(self):
        string = mr.String()
        self.assertTrue(isinstance(string._inner(), PyString))

        self.assertEqual("String", str(string))

        self.assertEqual("DataType.String", string.__repr__())

        self.assertEqual(mr.String(), mr.String())
        self.assertNotEqual(mr.String(), mr.Int())

    def test_int(self):
        integer = mr.Int()
        self.assertTrue(isinstance(integer._inner(), PyInt))

        self.assertEqual("Int", str(integer))

        self.assertEqual("DataType.Int", integer.__repr__())

        self.assertEqual(mr.Int(), mr.Int())
        self.assertNotEqual(mr.Int(), mr.String())

    def test_float(self):
        float = mr.Float()
        self.assertTrue(isinstance(float._inner(), PyFloat))

        self.assertEqual("Float", str(float))

        self.assertEqual("DataType.Float", float.__repr__())

        self.assertEqual(mr.Float(), mr.Float())
        self.assertNotEqual(mr.Float(), mr.String())

    def test_bool(self):
        bool = mr.Bool()
        self.assertTrue(isinstance(bool._inner(), PyBool))

        self.assertEqual("Bool", str(bool))

        self.assertEqual("DataType.Bool", bool.__repr__())

        self.assertEqual(mr.Bool(), mr.Bool())
        self.assertNotEqual(mr.Bool(), mr.String())

    def test_datetime(self):
        datetime = mr.DateTime()
        self.assertTrue(isinstance(datetime._inner(), PyDateTime))

        self.assertEqual("DateTime", str(datetime))

        self.assertEqual("DataType.DateTime", datetime.__repr__())

        self.assertEqual(mr.DateTime(), mr.DateTime())
        self.assertNotEqual(mr.DateTime(), mr.String())

    def test_null(self):
        null = mr.Null()
        self.assertTrue(isinstance(null._inner(), PyNull))

        self.assertEqual("Null", str(null))

        self.assertEqual("DataType.Null", null.__repr__())

        self.assertEqual(mr.Null(), mr.Null())
        self.assertNotEqual(mr.Null(), mr.String())

    def test_any(self):
        any = mr.Any()
        self.assertTrue(isinstance(any._inner(), PyAny))

        self.assertEqual("Any", str(any))

        self.assertEqual("DataType.Any", any.__repr__())

        self.assertEqual(mr.Any(), mr.Any())
        self.assertNotEqual(mr.Any(), mr.String())

    def test_union(self):
        union = mr.Union(mr.String(), mr.Int())
        self.assertTrue(isinstance(union._inner(), PyUnion))

        self.assertEqual("Union(String, Int)", str(union))

        self.assertEqual(
            "DataType.Union(DataType.String, DataType.Int)", union.__repr__()
        )

        union = mr.Union(mr.String(), mr.Int(), mr.Bool())
        self.assertTrue(isinstance(union._inner(), PyUnion))

        self.assertEqual("Union(String, Union(Int, Bool))", str(union))

        self.assertEqual(
            "DataType.Union(DataType.String, DataType.Union(DataType.Int, DataType.Bool))",
            union.__repr__(),
        )

        self.assertEqual(
            mr.Union(mr.String(), mr.Int()), mr.Union(mr.String(), mr.Int())
        )
        self.assertNotEqual(
            mr.Union(mr.String(), mr.Int()), mr.Union(mr.Int(), mr.String())
        )

    def test_invalid_union(self):
        with self.assertRaises(ValueError):
            mr.Union(mr.String())

    def test_option(self):
        option = mr.Option(mr.String())
        self.assertTrue(isinstance(option._inner(), PyOption))

        self.assertEqual("Option(String)", str(option))

        self.assertEqual("DataType.Option(DataType.String)", option.__repr__())

        self.assertEqual(mr.Option(mr.String()), mr.Option(mr.String()))
        self.assertNotEqual(mr.Option(mr.String()), mr.Option(mr.Int()))

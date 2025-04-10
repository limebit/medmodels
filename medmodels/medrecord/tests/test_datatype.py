import unittest

import medmodels.medrecord as mr
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
from medmodels.medrecord.datatype import DataType


class TestDataType(unittest.TestCase):
    def test_from_py_data_type(self) -> None:
        py_string = PyString()
        result = DataType._from_py_data_type(py_string)
        assert isinstance(result, mr.String)

        py_int = PyInt()
        result = DataType._from_py_data_type(py_int)
        assert isinstance(result, mr.Int)

        py_float = PyFloat()
        result = DataType._from_py_data_type(py_float)
        assert isinstance(result, mr.Float)

        py_bool = PyBool()
        result = DataType._from_py_data_type(py_bool)
        assert isinstance(result, mr.Bool)

        py_datetime = PyDateTime()
        result = DataType._from_py_data_type(py_datetime)
        assert isinstance(result, mr.DateTime)

        py_duration = PyDuration()
        result = DataType._from_py_data_type(py_duration)
        assert isinstance(result, mr.Duration)

        py_null = PyNull()
        result = DataType._from_py_data_type(py_null)
        assert isinstance(result, mr.Null)

        py_any = PyAny()
        result = DataType._from_py_data_type(py_any)
        assert isinstance(result, mr.Any)

        py_union = PyUnion(PyString(), PyInt())
        result = DataType._from_py_data_type(py_union)
        assert isinstance(result, mr.Union)
        assert result == mr.Union(mr.String(), mr.Int())

        nested_py_union = PyUnion(PyString(), PyUnion(PyInt(), PyBool()))
        result = DataType._from_py_data_type(nested_py_union)
        assert isinstance(result, mr.Union)
        assert result == mr.Union(mr.String(), mr.Union(mr.Int(), mr.Bool()))

        py_option = PyOption(PyString())
        result = DataType._from_py_data_type(py_option)
        assert isinstance(result, mr.Option)
        assert result == mr.Option(mr.String())

    def test_string(self) -> None:
        string = mr.String()
        assert isinstance(string._inner(), PyString)

        assert str(string) == "String"

        assert string.__repr__() == "DataType.String"

        assert mr.String() == mr.String()
        assert mr.String() != mr.Int()

    def test_int(self) -> None:
        integer = mr.Int()
        assert isinstance(integer._inner(), PyInt)

        assert str(integer) == "Int"

        assert integer.__repr__() == "DataType.Int"

        assert mr.Int() == mr.Int()
        assert mr.Int() != mr.String()

    def test_float(self) -> None:
        float = mr.Float()
        assert isinstance(float._inner(), PyFloat)

        assert str(float) == "Float"

        assert float.__repr__() == "DataType.Float"

        assert mr.Float() == mr.Float()
        assert mr.Float() != mr.String()

    def test_bool(self) -> None:
        bool = mr.Bool()
        assert isinstance(bool._inner(), PyBool)

        assert str(bool) == "Bool"

        assert bool.__repr__() == "DataType.Bool"

        assert mr.Bool() == mr.Bool()
        assert mr.Bool() != mr.String()

    def test_datetime(self) -> None:
        datetime = mr.DateTime()
        assert isinstance(datetime._inner(), PyDateTime)

        assert str(datetime) == "DateTime"

        assert datetime.__repr__() == "DataType.DateTime"

        assert mr.DateTime() == mr.DateTime()
        assert mr.DateTime() != mr.String()

    def test_duration(self) -> None:
        duration = mr.Duration()
        assert isinstance(duration._inner(), PyDuration)

        assert str(duration) == "Duration"

        assert duration.__repr__() == "DataType.Duration"

        assert mr.Duration() == mr.Duration()
        assert mr.Duration() != mr.String()

    def test_null(self) -> None:
        null = mr.Null()
        assert isinstance(null._inner(), PyNull)

        assert str(null) == "Null"

        assert null.__repr__() == "DataType.Null"

        assert mr.Null() == mr.Null()
        assert mr.Null() != mr.String()

    def test_any(self) -> None:
        any = mr.Any()
        assert isinstance(any._inner(), PyAny)

        assert str(any) == "Any"

        assert any.__repr__() == "DataType.Any"

        assert mr.Any() == mr.Any()
        assert mr.Any() != mr.String()

    def test_union(self) -> None:
        union = mr.Union(mr.String(), mr.Int())
        assert isinstance(union._inner(), PyUnion)

        assert str(union) == "Union(String, Int)"

        assert union.__repr__() == "DataType.Union(DataType.String, DataType.Int)"

        union = mr.Union(mr.String(), mr.Union(mr.Int(), mr.Bool()))
        assert isinstance(union._inner(), PyUnion)

        assert str(union) == "Union(String, Union(Int, Bool))"

        assert (
            union.__repr__()
            == "DataType.Union(DataType.String, DataType.Union(DataType.Int, DataType.Bool))"
        )

        assert mr.Union(mr.String(), mr.Int()) == mr.Union(mr.String(), mr.Int())
        assert mr.Union(mr.String(), mr.Int()) != mr.Union(mr.Int(), mr.String())

    def test_option(self) -> None:
        option = mr.Option(mr.String())
        assert isinstance(option._inner(), PyOption)

        assert str(option) == "Option(String)"

        assert option.__repr__() == "DataType.Option(DataType.String)"

        assert mr.Option(mr.String()) == mr.Option(mr.String())
        assert mr.Option(mr.String()) != mr.Option(mr.Int())


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestDataType)
    unittest.TextTestRunner(verbosity=2).run(run_test)

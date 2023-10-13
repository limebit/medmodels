use polars::prelude::AnyValue;
use pyo3::{IntoPy, PyObject, Python};
use pyo3_polars::PySeries;

pub struct PyAnyValue<'a>(pub AnyValue<'a>);

impl IntoPy<PyObject> for PyAnyValue<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            AnyValue::Binary(val) => val.into_py(py),
            AnyValue::Null => todo!(),
            AnyValue::Boolean(val) => val.into_py(py),
            AnyValue::Utf8(val) => val.into_py(py),
            AnyValue::UInt8(val) => val.into_py(py),
            AnyValue::UInt16(val) => val.into_py(py),
            AnyValue::UInt32(val) => val.into_py(py),
            AnyValue::UInt64(val) => val.into_py(py),
            AnyValue::Int8(val) => val.into_py(py),
            AnyValue::Int16(val) => val.into_py(py),
            AnyValue::Int32(val) => val.into_py(py),
            AnyValue::Int64(val) => val.into_py(py),
            AnyValue::Float32(val) => val.into_py(py),
            AnyValue::Float64(val) => val.into_py(py),
            AnyValue::Date(_) => todo!(),
            AnyValue::Datetime(_, _, _) => todo!(),
            AnyValue::Duration(_, _) => todo!(),
            AnyValue::Time(_) => todo!(),
            AnyValue::List(val) => PySeries(val).into_py(py),
            AnyValue::Utf8Owned(_) => todo!(),
            AnyValue::BinaryOwned(val) => val.into_py(py),
        }
    }
}

use polars::prelude::{AnyValue, TimeUnit};
use pyo3::{
    exceptions::PyTypeError,
    types::{PyDate, PyDateTime, PyDelta, PyNone, PyTime},
    FromPyObject, IntoPy, PyAny, PyErr, PyObject, PyResult, Python,
};
use pyo3_polars::PySeries;

macro_rules! convert_duration (
    ($py:expr, $difference:expr, $second_factor:literal) => {
        {
            let days = $difference / ($second_factor * 86400);
            let remaining_after_days = $difference % ($second_factor * 86400);
            let seconds = remaining_after_days / $second_factor;
            let remaining_after_seconds = remaining_after_days % $second_factor;
            let microseconds = remaining_after_seconds * (1000000 / $second_factor);

            PyDelta::new(
                $py,
                i32::try_from(days).unwrap(),
                i32::try_from(seconds).unwrap(),
                i32::try_from(microseconds).unwrap(),
                false,
            )
            .unwrap()
            .into_py($py)
        }
    }
);

pub struct PyAnyValue<'a>(pub AnyValue<'a>);

impl IntoPy<PyObject> for PyAnyValue<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            AnyValue::Binary(val) => val.into_py(py),
            AnyValue::Null => PyNone::get(py).into_py(py),
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
            AnyValue::Date(days) => PyDate::from_timestamp(py, (days * 86_400).into())
                .unwrap()
                .into_py(py),
            // The timezone is ignored - This may lead to wrong conversions
            AnyValue::Datetime(time, unit, _timezone) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000_000_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
            },
            AnyValue::Duration(difference, unit) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    convert_duration!(py, difference, 1_000)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    convert_duration!(py, difference, 1_000_000)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    convert_duration!(py, difference, 1_000_000_000)
                }
            },
            AnyValue::Time(nanoseconds) => {
                let hours = nanoseconds / 3_600_000_000_000;
                let remaining_after_hours = nanoseconds % 3_600_000_000_000;
                let minutes = remaining_after_hours / 60_000_000_000;
                let remaining_after_minutes = remaining_after_hours % 60_000_000_000;
                let seconds = remaining_after_minutes / 1_000_000_000;
                let remaining_after_seconds = remaining_after_minutes % 1_000_000_000;
                let microseconds = remaining_after_seconds / 1_000;

                PyTime::new(
                    py,
                    u8::try_from(hours).unwrap(),
                    u8::try_from(minutes).unwrap(),
                    u8::try_from(seconds).unwrap(),
                    u32::try_from(microseconds).unwrap(),
                    None,
                )
                .unwrap()
                .into_py(py)
            }
            AnyValue::List(val) => PySeries(val).into_py(py),
            AnyValue::Utf8Owned(val) => val.into_py(py),
            AnyValue::BinaryOwned(val) => val.into_py(py),
        }
    }
}

impl<'a> FromPyObject<'a> for PyAnyValue<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let object_type = ob
            .getattr("__class__")?
            .getattr("__name__")?
            .extract::<&str>()?;

        match object_type {
            "float" => Ok(PyAnyValue(AnyValue::Float64(ob.extract::<f64>()?))),
            "int" => Ok(PyAnyValue(AnyValue::Int64(ob.extract::<i64>()?))),
            "str" => Ok(PyAnyValue(AnyValue::Utf8(ob.extract::<&str>()?))),
            "bool" => Ok(PyAnyValue(AnyValue::Boolean(ob.extract::<bool>()?))),
            "datetime" => {
                let timestamp = (ob.call_method0("timestamp")?.extract::<f64>()? * 1_000.0) as i64;
                Ok(PyAnyValue(AnyValue::Datetime(
                    timestamp,
                    TimeUnit::Milliseconds,
                    &None,
                )))
            }
            "date" => {
                let days: Result<i32, PyErr> = Python::with_gil(|py| {
                    let datetime = py.import("datetime")?;

                    let epoch = datetime.call_method1("date", (1970, 1, 1))?;

                    let days = ob
                        .call_method1("__sub__", (epoch,))?
                        .getattr("days")?
                        .extract::<i32>()?;

                    Ok(days)
                });
                Ok(PyAnyValue(AnyValue::Date(days?)))
            }
            "timedelta" => {
                let seconds =
                    (ob.call_method0("total_seconds")?.extract::<f64>()? * 1_000.0) as i64;
                Ok(PyAnyValue(AnyValue::Duration(
                    seconds,
                    TimeUnit::Milliseconds,
                )))
            }
            "time" => {
                let hours = ob.getattr("hour")?.extract::<i64>()?;
                let minutes = ob.getattr("minute")?.extract::<i64>()?;
                let seconds = ob.getattr("second")?.extract::<i64>()?;
                let microseconds = ob.getattr("microsecond")?.extract::<i64>()?;

                Ok(PyAnyValue(AnyValue::Time(
                    (hours * 3_600_000_000_000)
                        + (minutes * 60_000_000_000)
                        + (seconds * 1_000_000_000)
                        + (microseconds * 1_000),
                )))
            }
            "Series" => Ok(PyAnyValue(AnyValue::List(ob.extract::<PySeries>()?.0))),
            "bytes" => Ok(PyAnyValue(AnyValue::Binary(ob.extract::<&[u8]>()?))),
            "NoneType" => Ok(PyAnyValue(AnyValue::Null)),
            _ => Err(PyTypeError::new_err(format!(
                "'{}' object cannot be interpreted",
                object_type
            ))),
        }
    }
}

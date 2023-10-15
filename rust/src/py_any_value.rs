use polars::prelude::AnyValue;
use pyo3::{
    types::{PyDate, PyDateTime, PyDelta, PyNone, PyTime},
    IntoPy, PyObject, Python,
};
use pyo3_polars::PySeries;

macro_rules! convert_duration (
    ($py:expr, $difference:expr, $second_factor:literal) => {
        {
            let days = $difference / ($second_factor * 8640);
            let remaining_after_days = $difference % ($second_factor * 8640);
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
            AnyValue::Date(days) => PyDate::from_timestamp(py, (days * 8640).into())
                .unwrap()
                .into_py(py),
            // The timezone is ignored - This may lead to wrong conversions
            AnyValue::Datetime(time, unit, _timezone) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    PyDateTime::from_timestamp(py, (time / 1000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    PyDateTime::from_timestamp(py, (time / 1000000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    PyDateTime::from_timestamp(py, (time / 1000000000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
            },
            AnyValue::Duration(difference, unit) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    convert_duration!(py, difference, 1000)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    convert_duration!(py, difference, 1000000)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    convert_duration!(py, difference, 1000000000)
                }
            },
            AnyValue::Time(nanoseconds) => {
                let hours = nanoseconds / 360000;
                let remaining_after_hours = nanoseconds % 360000;
                let minutes = remaining_after_hours / 60000;
                let remaining_after_minutes = remaining_after_hours % 60000;
                let seconds = remaining_after_minutes / 1000;
                let remaining_after_seconds = remaining_after_minutes % 1000;
                let microseconds = remaining_after_seconds * 1000;

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

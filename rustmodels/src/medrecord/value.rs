use super::{traits::DeepFrom, Lut};
use crate::{gil_hash_map::GILHashMap, medrecord::errors::PyMedRecordError};
use chrono::NaiveDateTime;
use medmodels_core::{errors::MedRecordError, medrecord::MedRecordValue};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyDateTime, PyDelta, PyFloat, PyInt, PyString},
    Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
};
use std::{ops::Deref, time::Duration};

#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyMedRecordValue(MedRecordValue);

impl From<MedRecordValue> for PyMedRecordValue {
    fn from(value: MedRecordValue) -> Self {
        Self(value)
    }
}

impl From<PyMedRecordValue> for MedRecordValue {
    fn from(value: PyMedRecordValue) -> Self {
        value.0
    }
}

impl DeepFrom<PyMedRecordValue> for MedRecordValue {
    fn deep_from(value: PyMedRecordValue) -> Self {
        value.into()
    }
}

impl DeepFrom<MedRecordValue> for PyMedRecordValue {
    fn deep_from(value: MedRecordValue) -> Self {
        value.into()
    }
}

impl Deref for PyMedRecordValue {
    type Target = MedRecordValue;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

static MEDRECORDVALUE_CONVERSION_LUT: Lut<MedRecordValue> = GILHashMap::new();

pub(crate) fn convert_pyobject_to_medrecordvalue(
    ob: &Bound<'_, PyAny>,
) -> PyResult<MedRecordValue> {
    fn convert_string(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::String(
            ob.extract::<String>().expect("Extraction must succeed"),
        ))
    }

    fn convert_int(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Int(
            ob.extract::<i64>().expect("Extraction must succeed"),
        ))
    }

    fn convert_float(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Float(
            ob.extract::<f64>().expect("Extraction must succeed"),
        ))
    }

    fn convert_bool(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Bool(
            ob.extract::<bool>().expect("Extraction must succeed"),
        ))
    }

    fn convert_datetime(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::DateTime(
            ob.extract::<NaiveDateTime>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_duration(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Duration(
            ob.extract::<Duration>().expect("Extraction must succeed"),
        ))
    }

    fn convert_null(_ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Null)
    }

    fn throw_error(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Err(
            PyMedRecordError::from(MedRecordError::ConversionError(format!(
                "Failed to convert {ob} into MedRecordValue",
            )))
            .into(),
        )
    }

    let type_pointer = ob.get_type_ptr() as usize;

    Python::with_gil(|py| {
        MEDRECORDVALUE_CONVERSION_LUT.map(py, |lut| {
            let conversion_function = lut.entry(type_pointer).or_insert_with(|| {
                if ob.is_instance_of::<PyString>() {
                    convert_string
                } else if ob.is_instance_of::<PyBool>() {
                    convert_bool
                } else if ob.is_instance_of::<PyInt>() {
                    convert_int
                } else if ob.is_instance_of::<PyFloat>() {
                    convert_float
                } else if ob.is_instance_of::<PyDateTime>() {
                    convert_datetime
                } else if ob.is_instance_of::<PyDelta>() {
                    convert_duration
                } else if ob.is_none() {
                    convert_null
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl FromPyObject<'_> for PyMedRecordValue {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        convert_pyobject_to_medrecordvalue(ob).map(PyMedRecordValue::from)
    }
}

impl<'py> IntoPyObject<'py> for PyMedRecordValue {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            MedRecordValue::String(value) => value.into_bound_py_any(py),
            MedRecordValue::Int(value) => value.into_bound_py_any(py),
            MedRecordValue::Float(value) => value.into_bound_py_any(py),
            MedRecordValue::Bool(value) => value.into_bound_py_any(py),
            MedRecordValue::DateTime(value) => value.into_bound_py_any(py),
            MedRecordValue::Duration(value) => value.into_bound_py_any(py),
            MedRecordValue::Null => py.None().into_bound_py_any(py),
        }
    }
}

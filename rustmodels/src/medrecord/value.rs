use super::{traits::DeepFrom, Lut};
use crate::{gil_hash_map::GILHashMap, medrecord::errors::PyMedRecordError};
use medmodels_core::{errors::MedRecordError, medrecord::MedRecordValue};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyFloat, PyInt, PyString},
    Bound, FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python,
};

#[repr(transparent)]
#[derive(Clone, Debug)]
pub(crate) struct PyMedRecordValue(MedRecordValue);

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

static MEDRECORDVALUE_CONVERSION_LUT: Lut<MedRecordValue> = GILHashMap::new();

pub(crate) fn convert_pyobject_to_medrecordvalue(
    ob: &Bound<'_, PyAny>,
) -> PyResult<MedRecordValue> {
    fn convert_string(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::String(ob.extract::<String>()?))
    }

    fn convert_int(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Int(ob.extract::<i64>()?))
    }

    fn convert_float(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Float(ob.extract::<f64>()?))
    }

    fn convert_bool(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Bool(ob.extract::<bool>()?))
    }

    fn convert_null(_ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Null)
    }

    fn throw_error(ob: &Bound<'_, PyAny>) -> PyResult<MedRecordValue> {
        Err(PyMedRecordError(MedRecordError::ConversionError(format!(
            "Failed to convert {} into MedRecordValue",
            ob,
        )))
        .into())
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

impl<'a> FromPyObject<'a> for PyMedRecordValue {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        convert_pyobject_to_medrecordvalue(ob).map(PyMedRecordValue::from)
    }
}

impl IntoPy<PyObject> for PyMedRecordValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            MedRecordValue::String(value) => value.into_py(py),
            MedRecordValue::Int(value) => value.into_py(py),
            MedRecordValue::Float(value) => value.into_py(py),
            MedRecordValue::Bool(value) => value.into_py(py),
            MedRecordValue::Null => py.None(),
        }
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

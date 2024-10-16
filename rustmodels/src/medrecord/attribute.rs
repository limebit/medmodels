use super::{traits::DeepFrom, value::convert_pyobject_to_medrecordvalue};
use crate::medrecord::errors::PyMedRecordError;
use medmodels_core::medrecord::MedRecordAttribute;
use pyo3::{Bound, FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python};
use std::{hash::Hash, ops::Deref};

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct PyMedRecordAttribute(MedRecordAttribute);

impl From<MedRecordAttribute> for PyMedRecordAttribute {
    fn from(value: MedRecordAttribute) -> Self {
        Self(value)
    }
}

impl From<PyMedRecordAttribute> for MedRecordAttribute {
    fn from(value: PyMedRecordAttribute) -> Self {
        value.0
    }
}

impl DeepFrom<PyMedRecordAttribute> for MedRecordAttribute {
    fn deep_from(value: PyMedRecordAttribute) -> Self {
        value.into()
    }
}

impl DeepFrom<MedRecordAttribute> for PyMedRecordAttribute {
    fn deep_from(value: MedRecordAttribute) -> Self {
        value.into()
    }
}

impl Deref for PyMedRecordAttribute {
    type Target = MedRecordAttribute;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PyMedRecordAttribute {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(convert_pyobject_to_medrecordvalue(ob)?
            .try_into()
            .map(|value: MedRecordAttribute| value.into())
            .map_err(PyMedRecordError::from)?)
    }
}

impl IntoPy<PyObject> for PyMedRecordAttribute {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            MedRecordAttribute::String(value) => value.into_py(py),
            MedRecordAttribute::Int(value) => value.into_py(py),
        }
    }
}

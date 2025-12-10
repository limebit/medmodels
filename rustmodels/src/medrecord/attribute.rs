use super::{traits::DeepFrom, value::convert_pyobject_to_medrecordvalue};
use crate::medrecord::errors::PyMedRecordError;
use medmodels::core::medrecord::MedRecordAttribute;
use pyo3::{Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python};
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

pub(crate) fn convert_pyobject_to_medrecordattribute(
    ob: &Bound<'_, PyAny>,
) -> PyResult<MedRecordAttribute> {
    Ok(convert_pyobject_to_medrecordvalue(ob)?
        .try_into()
        .map_err(PyMedRecordError::from)?)
}

impl FromPyObject<'_> for PyMedRecordAttribute {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        convert_pyobject_to_medrecordattribute(ob).map(PyMedRecordAttribute::from)
    }
}

impl<'py> IntoPyObject<'py> for PyMedRecordAttribute {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            MedRecordAttribute::String(value) => value.into_bound_py_any(py),
            MedRecordAttribute::Int(value) => value.into_bound_py_any(py),
        }
    }
}

use medmodels_core::errors::MedRecordError;
use pyo3::{
    exceptions::{PyAssertionError, PyIndexError, PyRuntimeError},
    PyErr,
};

pub(crate) struct PyMedRecordError(MedRecordError);

impl From<MedRecordError> for PyMedRecordError {
    fn from(error: MedRecordError) -> Self {
        Self(error)
    }
}

impl From<PyMedRecordError> for PyErr {
    fn from(error: PyMedRecordError) -> Self {
        match error.0 {
            MedRecordError::AssertionError(message) => PyAssertionError::new_err(message),
            MedRecordError::ConversionError(message) => PyRuntimeError::new_err(message),
            MedRecordError::IndexError(message) => PyIndexError::new_err(message),
        }
    }
}

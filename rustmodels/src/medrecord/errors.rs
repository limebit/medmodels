use medmodels_core::errors::{GraphError, MedRecordError};
use pyo3::{
    exceptions::{PyAssertionError, PyIndexError, PyKeyError, PyRuntimeError, PyValueError},
    PyErr,
};

#[repr(transparent)]
pub(crate) struct PyMedRecordError(MedRecordError);

impl From<MedRecordError> for PyMedRecordError {
    fn from(error: MedRecordError) -> Self {
        Self(error)
    }
}

impl From<GraphError> for PyMedRecordError {
    fn from(error: GraphError) -> Self {
        Self(MedRecordError::from(error))
    }
}

impl From<PyMedRecordError> for PyErr {
    fn from(error: PyMedRecordError) -> Self {
        match error.0 {
            MedRecordError::IndexError(message) => PyIndexError::new_err(message),
            MedRecordError::KeyError(message) => PyKeyError::new_err(message),
            MedRecordError::ConversionError(message) => PyRuntimeError::new_err(message),
            MedRecordError::AssertionError(message) => PyAssertionError::new_err(message),
            MedRecordError::SchemaError(message) => PyValueError::new_err(message),
            MedRecordError::QueryError(message) => PyRuntimeError::new_err(message),
        }
    }
}

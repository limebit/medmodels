use medmodels::core::MedRecord;
use pyo3::{
    types::{PyAnyMethods, PyBytes, PyBytesMethods},
    Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, Python,
};

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyMedRecord(pub MedRecord);

impl From<MedRecord> for PyMedRecord {
    fn from(value: MedRecord) -> Self {
        PyMedRecord(value)
    }
}

impl From<PyMedRecord> for MedRecord {
    fn from(value: PyMedRecord) -> Self {
        value.0
    }
}

impl AsRef<MedRecord> for PyMedRecord {
    fn as_ref(&self) -> &MedRecord {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for PyMedRecord {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let bytes = ob.call_method0("_to_bytes")?;
        let py_bytes: &Bound<'py, PyBytes> = bytes.downcast()?;

        let medrecord = bincode::deserialize(py_bytes.as_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self(medrecord))
    }
}

impl<'py> IntoPyObject<'py> for PyMedRecord {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let bytes = bincode::serialize(&self.0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let py_bytes = PyBytes::new(py, &bytes);

        let py_medrecord_class = py.import("medmodels._medmodels")?.getattr("PyMedRecord")?;
        let obj = py_medrecord_class.call_method1("_from_bytes", (py_bytes,))?;

        Ok(obj)
    }
}

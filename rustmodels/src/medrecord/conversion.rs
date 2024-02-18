use std::{collections::HashMap, hash::Hash};

use medmodels_core::{errors::MedRecordError, medrecord::MedRecordValue};
use pyo3::{FromPyObject, IntoPy, PyObject, PyResult};

use super::errors::PyMedRecordError;

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

impl<'a> FromPyObject<'a> for PyMedRecordValue {
    fn extract(ob: &'a pyo3::prelude::PyAny) -> PyResult<Self> {
        let object_type = ob
            .getattr("__class__")?
            .getattr("__name__")?
            .extract::<&str>()?;

        match object_type {
            "str" => Ok(MedRecordValue::String(ob.extract::<String>()?).into()),
            "int" => Ok(MedRecordValue::Int(ob.extract::<i64>()?).into()),
            "float" => Ok(MedRecordValue::Float(ob.extract::<f64>()?).into()),
            "bool" => Ok(MedRecordValue::Bool(ob.extract::<bool>()?).into()),
            _ => Err(
                Into::<PyMedRecordError>::into(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into MedRecordValue",
                    object_type
                )))
                .into(),
            ),
        }
    }
}

impl IntoPy<PyObject> for PyMedRecordValue {
    fn into_py(self, py: pyo3::prelude::Python<'_>) -> PyObject {
        match self.0 {
            MedRecordValue::String(value) => value.into_py(py),
            MedRecordValue::Int(value) => value.into_py(py),
            MedRecordValue::Float(value) => value.into_py(py),
            MedRecordValue::Bool(value) => value.into_py(py),
        }
    }
}

pub trait DeepFrom<T> {
    fn deep_from(value: T) -> Self;
}

pub trait DeepInto<T> {
    fn deep_into(self) -> T;
}

impl<T, F> DeepInto<T> for F
where
    T: DeepFrom<F>,
{
    fn deep_into(self) -> T {
        T::deep_from(self)
    }
}

impl<K, V, F> DeepFrom<HashMap<K, V>> for HashMap<K, F>
where
    K: Hash + Eq,
    F: From<V>,
{
    fn deep_from(value: HashMap<K, V>) -> Self {
        value
            .into_iter()
            .map(|(key, value)| (key, value.into()))
            .collect()
    }
}

impl<K, V, F> DeepFrom<(K, V)> for (K, F)
where
    F: DeepFrom<V>,
{
    fn deep_from(value: (K, V)) -> Self {
        (value.0, value.1.deep_into())
    }
}

impl<K, K2, V, F> DeepFrom<(K, K2, V)> for (K, K2, F)
where
    F: DeepFrom<V>,
{
    fn deep_from(value: (K, K2, V)) -> Self {
        (value.0, value.1, value.2.deep_into())
    }
}

impl<V, F> DeepFrom<Vec<V>> for Vec<F>
where
    F: DeepFrom<V>,
{
    fn deep_from(value: Vec<V>) -> Self {
        value.into_iter().map(|value| value.deep_into()).collect()
    }
}

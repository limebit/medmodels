use std::{collections::HashMap, hash::Hash};

use super::errors::PyMedRecordError;
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{MedRecordAttribute, MedRecordValue},
};
use pyo3::{FromPyObject, IntoPy, PyObject, PyResult};

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
            _ => Err(PyMedRecordError(MedRecordError::ConversionError(format!(
                "Failed to convert {} into MedRecordValue",
                object_type
            )))
            .into()),
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

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct PyMedRecordAttribute(MedRecordAttribute);

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

impl<'a> FromPyObject<'a> for PyMedRecordAttribute {
    fn extract(ob: &'a pyo3::prelude::PyAny) -> PyResult<Self> {
        let object_type = ob
            .getattr("__class__")?
            .getattr("__name__")?
            .extract::<&str>()?;

        match object_type {
            "str" => Ok(MedRecordAttribute::String(ob.extract::<String>()?).into()),
            "int" => Ok(MedRecordAttribute::Int(ob.extract::<i64>()?).into()),
            _ => Err(PyMedRecordError(MedRecordError::ConversionError(format!(
                "Failed to convert {} into MedRecordValue",
                object_type
            )))
            .into()),
        }
    }
}

impl IntoPy<PyObject> for PyMedRecordAttribute {
    fn into_py(self, py: pyo3::prelude::Python<'_>) -> PyObject {
        match self.0 {
            MedRecordAttribute::String(value) => value.into_py(py),
            MedRecordAttribute::Int(value) => value.into_py(py),
        }
    }
}

pub trait DeepFrom<T> {
    fn deep_from(value: T) -> Self;
}

pub trait DeepInto<T> {
    fn deep_into(self) -> T;
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

impl<T, F> DeepInto<T> for F
where
    T: DeepFrom<F>,
{
    fn deep_into(self) -> T {
        T::deep_from(self)
    }
}

impl<K, KF, V, VF> DeepFrom<HashMap<K, V>> for HashMap<KF, VF>
where
    KF: Hash + Eq + DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: HashMap<K, V>) -> Self {
        value
            .into_iter()
            .map(|(key, value)| (key.deep_into(), value.deep_into()))
            .collect()
    }
}

impl<K, KF, V, VF> DeepFrom<(K, V)> for (KF, VF)
where
    KF: DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: (K, V)) -> Self {
        (value.0.deep_into(), value.1.deep_into())
    }
}

impl<K, KF, V, VF> DeepFrom<(K, K, V)> for (KF, KF, VF)
where
    KF: DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: (K, K, V)) -> Self {
        (
            value.0.deep_into(),
            value.1.deep_into(),
            value.2.deep_into(),
        )
    }
}

impl<V, VF> DeepFrom<Vec<V>> for Vec<VF>
where
    VF: DeepFrom<V>,
{
    fn deep_from(value: Vec<V>) -> Self {
        value.into_iter().map(VF::deep_from).collect()
    }
}

impl<V, VF> DeepFrom<Option<V>> for Option<VF>
where
    VF: DeepFrom<V>,
{
    fn deep_from(value: Option<V>) -> Self {
        value.map(|v| v.deep_into())
    }
}

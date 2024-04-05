use crate::{gil_hash_map::GILHashMap, medrecord::errors::PyMedRecordError};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{MedRecordAttribute, MedRecordValue},
};
use pyo3::{
    types::{PyBool, PyFloat, PyInt, PyString, PyType},
    FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python,
};
use std::{collections::HashMap, hash::Hash};

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

static MEDRECORDVALUE_CONVERSION_LUT: GILHashMap<usize, fn(&PyAny) -> PyResult<MedRecordValue>> =
    GILHashMap::new();

pub fn convert_pyobject_to_medrecordvalue(ob: &PyAny) -> PyResult<MedRecordValue> {
    fn convert_string(ob: &PyAny) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::String(ob.extract::<String>()?))
    }

    fn convert_int(ob: &PyAny) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Int(ob.extract::<i64>()?))
    }

    fn convert_float(ob: &PyAny) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Float(ob.extract::<f64>()?))
    }

    fn convert_bool(ob: &PyAny) -> PyResult<MedRecordValue> {
        Ok(MedRecordValue::Bool(ob.extract::<bool>()?))
    }

    fn throw_error(ob: &PyAny) -> PyResult<MedRecordValue> {
        Err(PyMedRecordError(MedRecordError::ConversionError(format!(
            "Failed to convert {} into MedRecordValue",
            ob,
        )))
        .into())
    }

    let type_pointer = PyType::as_type_ptr(ob.get_type()) as usize;

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
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl<'a> FromPyObject<'a> for PyMedRecordValue {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        convert_pyobject_to_medrecordvalue(ob).map(PyMedRecordValue::from)
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

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
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
        Ok(convert_pyobject_to_medrecordvalue(ob)?
            .try_into()
            .map(|value: MedRecordAttribute| value.into())
            .map_err(PyMedRecordError::from)?)
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

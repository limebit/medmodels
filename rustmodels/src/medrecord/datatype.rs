#![allow(clippy::new_without_default)]

use super::{traits::DeepFrom, Lut};
use crate::{gil_hash_map::GILHashMap, medrecord::errors::PyMedRecordError};
use medmodels_core::{errors::MedRecordError, medrecord::datatypes::DataType};
use pyo3::{prelude::*, IntoPyObjectExt};

macro_rules! implement_pymethods {
    ($struct:ty) => {
        #[pymethods]
        impl $struct {
            #[new]
            pub fn new() -> Self {
                Self {}
            }
        }
    };
}

#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyDataType(DataType);

impl From<DataType> for PyDataType {
    fn from(value: DataType) -> Self {
        Self(value)
    }
}

impl From<PyDataType> for DataType {
    fn from(value: PyDataType) -> Self {
        value.0
    }
}

impl DeepFrom<PyDataType> for DataType {
    fn deep_from(value: PyDataType) -> Self {
        value.into()
    }
}

impl DeepFrom<DataType> for PyDataType {
    fn deep_from(value: DataType) -> Self {
        value.into()
    }
}

static DATATYPE_CONVERSION_LUT: Lut<DataType> = GILHashMap::new();

pub(crate) fn convert_pyobject_to_datatype(ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
    fn convert_string(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::String)
    }

    fn convert_int(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Int)
    }

    fn convert_float(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Float)
    }

    fn convert_bool(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Bool)
    }

    fn convert_datetime(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::DateTime)
    }

    fn convert_duration(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Duration)
    }

    fn convert_null(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Null)
    }

    fn convert_any(_ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Ok(DataType::Any)
    }

    fn convert_union(ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        let union = ob
            .extract::<PyRef<PyUnion>>()
            .expect("Extraction must succeed");

        let dtypes = union.0.clone();

        Ok(DataType::Union((
            Box::new(dtypes.0.into()),
            Box::new(dtypes.1.into()),
        )))
    }

    fn convert_option(ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        let option = ob
            .extract::<PyRef<PyOption>>()
            .expect("Extraction must succeed");

        Ok(DataType::Option(Box::new(option.0.clone().into())))
    }

    fn throw_error(ob: &Bound<'_, pyo3::PyAny>) -> PyResult<DataType> {
        Err(
            PyMedRecordError::from(MedRecordError::ConversionError(format!(
                "Failed to convert {ob} into DataType",
            )))
            .into(),
        )
    }

    let type_pointer = ob.get_type_ptr() as usize;

    Python::with_gil(|py| {
        DATATYPE_CONVERSION_LUT.map(py, |lut| {
            let conversion_function = lut.entry(type_pointer).or_insert_with(|| {
                if ob.is_instance_of::<PyString>() {
                    convert_string
                } else if ob.is_instance_of::<PyInt>() {
                    convert_int
                } else if ob.is_instance_of::<PyFloat>() {
                    convert_float
                } else if ob.is_instance_of::<PyBool>() {
                    convert_bool
                } else if ob.is_instance_of::<PyDateTime>() {
                    convert_datetime
                } else if ob.is_instance_of::<PyDuration>() {
                    convert_duration
                } else if ob.is_instance_of::<PyNull>() {
                    convert_null
                } else if ob.is_instance_of::<PyAny>() {
                    convert_any
                } else if ob.is_instance_of::<PyUnion>() {
                    convert_union
                } else if ob.is_instance_of::<PyOption>() {
                    convert_option
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl FromPyObject<'_> for PyDataType {
    fn extract_bound(ob: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        convert_pyobject_to_datatype(ob).map(Self)
    }
}

impl<'py> IntoPyObject<'py> for PyDataType {
    type Target = pyo3::PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            DataType::String => PyString {}.into_bound_py_any(py),
            DataType::Int => PyInt {}.into_bound_py_any(py),
            DataType::Float => PyFloat {}.into_bound_py_any(py),
            DataType::Bool => PyBool {}.into_bound_py_any(py),
            DataType::DateTime => PyDateTime {}.into_bound_py_any(py),
            DataType::Duration => PyDuration {}.into_bound_py_any(py),
            DataType::Null => PyNull {}.into_bound_py_any(py),
            DataType::Any => PyAny {}.into_bound_py_any(py),
            DataType::Union((dtype1, dtype2)) => {
                PyUnion(((*dtype1).into(), (*dtype2).into())).into_bound_py_any(py)
            }
            DataType::Option(dtype) => PyOption((*dtype).into()).into_bound_py_any(py),
        }
    }
}

#[pyclass]
pub struct PyString;
implement_pymethods!(PyString);

#[pyclass]
pub struct PyInt;
implement_pymethods!(PyInt);

#[pyclass]
pub struct PyFloat;
implement_pymethods!(PyFloat);

#[pyclass]
pub struct PyBool;
implement_pymethods!(PyBool);

#[pyclass]
pub struct PyDateTime;
implement_pymethods!(PyDateTime);

#[pyclass]
pub struct PyDuration;
implement_pymethods!(PyDuration);

#[pyclass]
pub struct PyNull;
implement_pymethods!(PyNull);

#[pyclass]
pub struct PyAny;
implement_pymethods!(PyAny);

#[pyclass]
pub struct PyUnion((PyDataType, PyDataType));

#[pymethods]
impl PyUnion {
    #[new]
    fn new(dtype1: PyDataType, dtype2: PyDataType) -> Self {
        Self((dtype1, dtype2))
    }

    #[getter]
    fn dtype1(&self) -> PyDataType {
        self.0 .0.clone()
    }

    #[getter]
    fn dtype2(&self) -> PyDataType {
        self.0 .1.clone()
    }
}

#[pyclass]
pub struct PyOption(PyDataType);

#[pymethods]
impl PyOption {
    #[new]
    fn new(dtype: PyDataType) -> Self {
        Self(dtype)
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.0.clone()
    }
}

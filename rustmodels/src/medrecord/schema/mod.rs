mod inferred;
mod provided;

pub use self::{
    inferred::{PyInferredGroupSchema, PyInferredSchema},
    provided::{PyProvidedGroupSchema, PyProvidedSchema},
};
use super::{datatype::PyDataType, traits::DeepFrom, Lut};
use crate::{gil_hash_map::GILHashMap, medrecord::errors::PyMedRecordError};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{AttributeDataType, AttributeType, Schema},
};
use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyAttributeType {
    Categorical = 0,
    Continuous = 1,
    Temporal = 2,
    Unstructured = 3,
}

impl From<AttributeType> for PyAttributeType {
    fn from(value: AttributeType) -> Self {
        match value {
            AttributeType::Categorical => Self::Categorical,
            AttributeType::Continuous => Self::Continuous,
            AttributeType::Temporal => Self::Temporal,
            AttributeType::Unstructured => Self::Unstructured,
        }
    }
}

impl From<PyAttributeType> for AttributeType {
    fn from(value: PyAttributeType) -> Self {
        match value {
            PyAttributeType::Categorical => Self::Categorical,
            PyAttributeType::Continuous => Self::Continuous,
            PyAttributeType::Temporal => Self::Temporal,
            PyAttributeType::Unstructured => Self::Unstructured,
        }
    }
}

#[pymethods]
impl PyAttributeType {
    #[staticmethod]
    fn infer_from(data_type: PyDataType) -> Self {
        AttributeType::infer_from(&data_type.into()).into()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyAttributeDataType {
    data_type: PyDataType,
    attribute_type: PyAttributeType,
}

impl From<PyAttributeDataType> for AttributeDataType {
    fn from(value: PyAttributeDataType) -> Self {
        Self {
            data_type: value.data_type.into(),
            attribute_type: value.attribute_type.into(),
        }
    }
}

impl From<AttributeDataType> for PyAttributeDataType {
    fn from(value: AttributeDataType) -> Self {
        Self {
            data_type: value.data_type.into(),
            attribute_type: value.attribute_type.into(),
        }
    }
}

impl DeepFrom<PyAttributeDataType> for AttributeDataType {
    fn deep_from(value: PyAttributeDataType) -> AttributeDataType {
        value.into()
    }
}

impl DeepFrom<AttributeDataType> for PyAttributeDataType {
    fn deep_from(value: AttributeDataType) -> PyAttributeDataType {
        value.into()
    }
}

#[pymethods]
impl PyAttributeDataType {
    #[new]
    #[pyo3(signature = (data_type, attribute_type))]
    pub fn new(data_type: PyDataType, attribute_type: PyAttributeType) -> Self {
        Self {
            data_type,
            attribute_type,
        }
    }

    #[getter]
    pub fn data_type(&self) -> PyDataType {
        self.data_type.clone()
    }

    #[getter]
    pub fn attribute_type(&self) -> PyAttributeType {
        self.attribute_type.clone()
    }
}

pub struct PySchema(Schema);

impl From<PySchema> for Schema {
    fn from(value: PySchema) -> Self {
        value.0
    }
}

impl From<Schema> for PySchema {
    fn from(value: Schema) -> Self {
        Self(value)
    }
}

impl FromPyObject<'_> for PySchema {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(convert_pyobject_to_datatype(ob)?.into())
    }
}

static DATATYPE_CONVERSION_LUT: Lut<Schema> = GILHashMap::new();

fn convert_pyobject_to_datatype(ob: &Bound<'_, PyAny>) -> PyResult<Schema> {
    fn convert_inferred(ob: &Bound<'_, PyAny>) -> PyResult<Schema> {
        Ok(Schema::Inferred(ob.extract::<PyInferredSchema>()?.into()))
    }

    fn convert_provided(ob: &Bound<'_, PyAny>) -> PyResult<Schema> {
        Ok(Schema::Provided(ob.extract::<PyProvidedSchema>()?.into()))
    }

    fn throw_error(ob: &Bound<'_, PyAny>) -> PyResult<Schema> {
        Err(
            PyMedRecordError::from(MedRecordError::ConversionError(format!(
                "Failed to convert {} into Schema",
                ob,
            )))
            .into(),
        )
    }

    let type_pointer = ob.get_type_ptr() as usize;

    Python::with_gil(|py| {
        DATATYPE_CONVERSION_LUT.map(py, |lut| {
            let conversion_function = lut.entry(type_pointer).or_insert_with(|| {
                if ob.is_instance_of::<PyInferredSchema>() {
                    convert_inferred
                } else if ob.is_instance_of::<PyProvidedSchema>() {
                    convert_provided
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl IntoPy<PyObject> for PySchema {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            Schema::Inferred(value) => PyInferredSchema::from(value).into_py(py),
            Schema::Provided(value) => PyProvidedSchema::from(value).into_py(py),
        }
    }
}

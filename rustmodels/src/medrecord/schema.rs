use medmodels_core::{
    errors::MedRecordError,
    medrecord::{AttributeDataType, AttributeType, ProvidedGroupSchema, ProvidedSchema, Schema},
};
use pyo3::prelude::*;
use std::collections::HashMap;

use super::{
    attribute::PyMedRecordAttribute,
    datatype::PyDataType,
    errors::PyMedRecordError,
    traits::{DeepFrom, DeepInto},
    PyGroup,
};

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

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyGroupSchema(ProvidedGroupSchema);

impl From<ProvidedGroupSchema> for PyGroupSchema {
    fn from(value: ProvidedGroupSchema) -> Self {
        Self(value)
    }
}

impl From<PyGroupSchema> for ProvidedGroupSchema {
    fn from(value: PyGroupSchema) -> Self {
        value.0
    }
}

impl DeepFrom<PyGroupSchema> for ProvidedGroupSchema {
    fn deep_from(value: PyGroupSchema) -> ProvidedGroupSchema {
        value.into()
    }
}

impl DeepFrom<ProvidedGroupSchema> for PyGroupSchema {
    fn deep_from(value: ProvidedGroupSchema) -> PyGroupSchema {
        value.into()
    }
}

#[pymethods]
impl PyGroupSchema {
    #[new]
    #[pyo3(signature = (nodes, edges, strict=true))]
    fn new(
        nodes: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        edges: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        strict: bool,
    ) -> Self {
        PyGroupSchema(ProvidedGroupSchema {
            nodes: nodes.deep_into(),
            edges: edges.deep_into(),
            strict,
        })
    }

    #[getter]
    fn nodes(&self) -> HashMap<PyMedRecordAttribute, PyAttributeDataType> {
        self.0.nodes.clone().deep_into()
    }

    #[getter]
    fn edges(&self) -> HashMap<PyMedRecordAttribute, PyAttributeDataType> {
        self.0.edges.clone().deep_into()
    }

    #[getter]
    fn strict(&self) -> bool {
        self.0.strict
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PySchema(Schema);

impl From<Schema> for PySchema {
    fn from(value: Schema) -> Self {
        Self(value)
    }
}

impl From<PySchema> for Schema {
    fn from(value: PySchema) -> Self {
        value.0
    }
}

#[pymethods]
impl PySchema {
    #[getter]
    fn groups(&self) -> Vec<PyGroup> {
        match self.0 {
            Schema::Provided(ref schema) => {
                schema.groups.keys().map(|g| g.clone().into()).collect()
            }
            Schema::Inferred(ref schema) => {
                schema.groups.keys().map(|g| g.clone().into()).collect()
            }
        }
    }

    fn group(&self, group: PyGroup) -> PyResult<PyGroupSchema> {
        let group = group.into();

        Ok(self
            .0
            .groups
            .get(&group)
            .map(|g| g.clone().into())
            .ok_or(PyMedRecordError::from(MedRecordError::SchemaError(
                format!("No schema found for group: {}", group),
            )))?)
    }

    #[getter]
    fn default(&self) -> PyGroupSchema {
        self.0.default.clone().into()
    }
}

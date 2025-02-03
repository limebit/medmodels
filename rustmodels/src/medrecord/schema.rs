use medmodels_core::{
    errors::MedRecordError,
    medrecord::{AttributeDataType, AttributeType, GroupSchema, InferredSchema},
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
}

impl From<AttributeType> for PyAttributeType {
    fn from(value: AttributeType) -> Self {
        match value {
            AttributeType::Categorical => Self::Categorical,
            AttributeType::Continuous => Self::Continuous,
            AttributeType::Temporal => Self::Temporal,
        }
    }
}

impl From<PyAttributeType> for AttributeType {
    fn from(value: PyAttributeType) -> Self {
        match value {
            PyAttributeType::Categorical => Self::Categorical,
            PyAttributeType::Continuous => Self::Continuous,
            PyAttributeType::Temporal => Self::Temporal,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyAttributeDataType {
    data_type: PyDataType,
    attribute_type: Option<PyAttributeType>,
}

impl From<PyAttributeDataType> for AttributeDataType {
    fn from(value: PyAttributeDataType) -> Self {
        Self {
            data_type: value.data_type.into(),
            attribute_type: value.attribute_type.map(|t| t.into()),
        }
    }
}

impl From<AttributeDataType> for PyAttributeDataType {
    fn from(value: AttributeDataType) -> Self {
        Self {
            data_type: value.data_type.into(),
            attribute_type: value.attribute_type.map(|t| t.into()),
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
    #[pyo3(signature = (data_type, attribute_type=None))]
    pub fn new(data_type: PyDataType, attribute_type: Option<PyAttributeType>) -> Self {
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
    pub fn attribute_type(&self) -> Option<PyAttributeType> {
        self.attribute_type.clone()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyGroupSchema(GroupSchema);

impl From<GroupSchema> for PyGroupSchema {
    fn from(value: GroupSchema) -> Self {
        Self(value)
    }
}

impl From<PyGroupSchema> for GroupSchema {
    fn from(value: PyGroupSchema) -> Self {
        value.0
    }
}

impl DeepFrom<PyGroupSchema> for GroupSchema {
    fn deep_from(value: PyGroupSchema) -> GroupSchema {
        value.into()
    }
}

impl DeepFrom<GroupSchema> for PyGroupSchema {
    fn deep_from(value: GroupSchema) -> PyGroupSchema {
        value.into()
    }
}

#[pymethods]
impl PyGroupSchema {
    #[new]
    #[pyo3(signature = (nodes, edges, strict=None))]
    fn new(
        nodes: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        edges: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        strict: Option<bool>,
    ) -> Self {
        PyGroupSchema(GroupSchema {
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
    fn strict(&self) -> Option<bool> {
        self.0.strict
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PySchema(InferredSchema);

impl From<InferredSchema> for PySchema {
    fn from(value: InferredSchema) -> Self {
        Self(value)
    }
}

impl From<PySchema> for InferredSchema {
    fn from(value: PySchema) -> Self {
        value.0
    }
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (groups, default=None, strict=None))]
    fn new(
        groups: HashMap<PyGroup, PyGroupSchema>,
        default: Option<PyGroupSchema>,
        strict: Option<bool>,
    ) -> Self {
        PySchema(InferredSchema {
            groups: groups
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            default: default.deep_into(),
            strict,
        })
    }

    #[getter]
    fn groups(&self) -> Vec<PyGroup> {
        self.0.groups.keys().map(|g| g.clone().into()).collect()
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
    fn default(&self) -> Option<PyGroupSchema> {
        self.0.default.clone().map(|g| g.into())
    }

    #[getter]
    fn strict(&self) -> Option<bool> {
        self.0.strict
    }
}

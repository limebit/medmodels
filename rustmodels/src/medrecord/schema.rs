use medmodels_core::medrecord::{GroupSchema, Schema};
use pyo3::prelude::*;
use std::collections::HashMap;

use super::{
    attribute::PyMedRecordAttribute,
    datatype::PyDataType,
    traits::{DeepFrom, DeepInto},
    PyGroup,
};

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
    fn new(
        nodes: HashMap<PyMedRecordAttribute, PyDataType>,
        edges: HashMap<PyMedRecordAttribute, PyDataType>,
        strict: Option<bool>,
    ) -> Self {
        PyGroupSchema(GroupSchema {
            nodes: nodes.deep_into(),
            edges: edges.deep_into(),
            strict,
        })
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
    #[new]
    fn new(
        groups: HashMap<PyGroup, PyGroupSchema>,
        default: Option<PyGroupSchema>,
        strict: Option<bool>,
    ) -> Self {
        PySchema(Schema {
            groups: groups
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            default: default.deep_into(),
            strict,
        })
    }
}

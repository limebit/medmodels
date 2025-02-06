use super::PyAttributeDataType;
use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    errors::PyMedRecordError,
    traits::{DeepFrom, DeepInto},
    PyGroup,
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{InferredGroupSchema, InferredSchema},
};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyInferredGroupSchema(InferredGroupSchema);

impl From<InferredGroupSchema> for PyInferredGroupSchema {
    fn from(value: InferredGroupSchema) -> Self {
        Self(value)
    }
}

impl From<PyInferredGroupSchema> for InferredGroupSchema {
    fn from(value: PyInferredGroupSchema) -> Self {
        value.0
    }
}

impl DeepFrom<PyInferredGroupSchema> for InferredGroupSchema {
    fn deep_from(value: PyInferredGroupSchema) -> InferredGroupSchema {
        value.into()
    }
}

impl DeepFrom<InferredGroupSchema> for PyInferredGroupSchema {
    fn deep_from(value: InferredGroupSchema) -> PyInferredGroupSchema {
        value.into()
    }
}

#[pymethods]
impl PyInferredGroupSchema {
    #[new]
    #[pyo3(signature = (nodes, edges))]
    fn new(
        nodes: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        edges: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
    ) -> Self {
        PyInferredGroupSchema(InferredGroupSchema {
            nodes: nodes.deep_into(),
            edges: edges.deep_into(),
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
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyInferredSchema(InferredSchema);

impl From<InferredSchema> for PyInferredSchema {
    fn from(value: InferredSchema) -> Self {
        Self(value)
    }
}

impl From<PyInferredSchema> for InferredSchema {
    fn from(value: PyInferredSchema) -> Self {
        value.0
    }
}

#[pymethods]
impl PyInferredSchema {
    #[getter]
    fn groups(&self) -> Vec<PyGroup> {
        self.0.groups.keys().map(|g| g.clone().into()).collect()
    }

    fn group(&self, group: PyGroup) -> PyResult<PyInferredGroupSchema> {
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
    fn default(&self) -> PyInferredGroupSchema {
        self.0.default.clone().into()
    }
}

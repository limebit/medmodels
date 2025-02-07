use super::{PyAttributeDataType, PyInferredGroupSchema};
use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    errors::PyMedRecordError,
    traits::{DeepFrom, DeepInto},
    PyGroup,
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{InferredGroupSchema, ProvidedGroupSchema, ProvidedSchema},
};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyProvidedGroupSchema(ProvidedGroupSchema);

impl From<ProvidedGroupSchema> for PyProvidedGroupSchema {
    fn from(value: ProvidedGroupSchema) -> Self {
        Self(value)
    }
}

impl From<PyProvidedGroupSchema> for ProvidedGroupSchema {
    fn from(value: PyProvidedGroupSchema) -> Self {
        value.0
    }
}

impl DeepFrom<PyProvidedGroupSchema> for ProvidedGroupSchema {
    fn deep_from(value: PyProvidedGroupSchema) -> ProvidedGroupSchema {
        value.into()
    }
}

impl DeepFrom<ProvidedGroupSchema> for PyProvidedGroupSchema {
    fn deep_from(value: ProvidedGroupSchema) -> PyProvidedGroupSchema {
        value.into()
    }
}

#[pymethods]
impl PyProvidedGroupSchema {
    #[new]
    #[pyo3(signature = (nodes, edges))]
    fn new(
        nodes: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
        edges: HashMap<PyMedRecordAttribute, PyAttributeDataType>,
    ) -> Self {
        PyProvidedGroupSchema(ProvidedGroupSchema {
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

    fn _into_py_inferred_group_schema(&self) -> PyInferredGroupSchema {
        InferredGroupSchema::from(self.0.clone()).into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyProvidedSchema(ProvidedSchema);

impl From<ProvidedSchema> for PyProvidedSchema {
    fn from(value: ProvidedSchema) -> Self {
        Self(value)
    }
}

impl From<PyProvidedSchema> for ProvidedSchema {
    fn from(value: PyProvidedSchema) -> Self {
        value.0
    }
}

#[pymethods]
impl PyProvidedSchema {
    #[new]
    #[pyo3(signature = (groups, default))]
    fn new(
        groups: HashMap<PyGroup, PyProvidedGroupSchema>,
        default: PyProvidedGroupSchema,
    ) -> Self {
        PyProvidedSchema(ProvidedSchema {
            groups: groups
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            default: default.deep_into(),
        })
    }

    #[getter]
    fn groups(&self) -> Vec<PyGroup> {
        self.0.groups.keys().map(|g| g.clone().into()).collect()
    }

    fn group(&self, group: PyGroup) -> PyResult<PyProvidedGroupSchema> {
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
    fn default(&self) -> PyProvidedGroupSchema {
        self.0.default.clone().into()
    }
}

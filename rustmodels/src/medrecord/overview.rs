use std::collections::HashMap;

use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    datatype::PyDataType,
    schema::PyAttributeType,
    traits::{DeepFrom, DeepInto},
    value::PyMedRecordValue,
};
use medmodels::core::medrecord::overview::{
    AttributeOverview, AttributeOverviewData, EdgeGroupOverview, GroupOverview, NodeGroupOverview,
    Overview,
};
use pyo3::{prelude::*, types::PyDict};

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyAttributeOverview(AttributeOverview);

impl From<AttributeOverview> for PyAttributeOverview {
    fn from(value: AttributeOverview) -> Self {
        Self(value)
    }
}

impl From<PyAttributeOverview> for AttributeOverview {
    fn from(value: PyAttributeOverview) -> Self {
        value.0
    }
}

impl DeepFrom<AttributeOverview> for PyAttributeOverview {
    fn deep_from(value: AttributeOverview) -> Self {
        value.into()
    }
}

impl DeepFrom<PyAttributeOverview> for AttributeOverview {
    fn deep_from(value: PyAttributeOverview) -> Self {
        value.into()
    }
}

impl AsRef<AttributeOverview> for PyAttributeOverview {
    fn as_ref(&self) -> &AttributeOverview {
        &self.0
    }
}

#[pymethods]
impl PyAttributeOverview {
    #[getter]
    pub fn data_type(&self) -> PyDataType {
        self.0.data_type.clone().into()
    }

    #[getter]
    pub fn data(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);

        match &self.0.data {
            AttributeOverviewData::Categorical { distinct_values } => {
                let distinct_values: Vec<PyMedRecordValue> = distinct_values.clone().deep_into();

                dict.set_item("distinct_values", distinct_values)
                    .expect("Setting item must succeed");
                dict.set_item("attribute_type", PyAttributeType::Categorical)
                    .expect("Setting item must succeed");

                dict.into_pyobject(py)
                    .expect("Conversion must succeed")
                    .into()
            }
            AttributeOverviewData::Continuous { min, mean, max } => {
                dict.set_item("min", PyMedRecordValue::from(min.clone()))
                    .expect("Setting item must succeed");
                dict.set_item("mean", PyMedRecordValue::from(mean.clone()))
                    .expect("Setting item must succeed");
                dict.set_item("max", PyMedRecordValue::from(max.clone()))
                    .expect("Setting item must succeed");
                dict.set_item("attribute_type", PyAttributeType::Continuous)
                    .expect("Setting item must succeed");

                dict.into()
            }
            AttributeOverviewData::Temporal { min, max } => {
                dict.set_item("min", PyMedRecordValue::from(min.clone()))
                    .expect("Setting item must succeed");
                dict.set_item("max", PyMedRecordValue::from(max.clone()))
                    .expect("Setting item must succeed");
                dict.set_item("attribute_type", PyAttributeType::Temporal)
                    .expect("Setting item must succeed");

                dict.into()
            }
            AttributeOverviewData::Unstructured { distinct_count } => {
                dict.set_item("distinct_count", *distinct_count)
                    .expect("Setting item must succeed");
                dict.set_item("attribute_type", PyAttributeType::Unstructured)
                    .expect("Setting item must succeed");

                dict.into()
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyNodeGroupOverview(NodeGroupOverview);

impl From<NodeGroupOverview> for PyNodeGroupOverview {
    fn from(value: NodeGroupOverview) -> Self {
        Self(value)
    }
}

impl From<PyNodeGroupOverview> for NodeGroupOverview {
    fn from(value: PyNodeGroupOverview) -> Self {
        value.0
    }
}

impl DeepFrom<NodeGroupOverview> for PyNodeGroupOverview {
    fn deep_from(value: NodeGroupOverview) -> Self {
        value.into()
    }
}

impl DeepFrom<PyNodeGroupOverview> for NodeGroupOverview {
    fn deep_from(value: PyNodeGroupOverview) -> Self {
        value.into()
    }
}

impl AsRef<NodeGroupOverview> for PyNodeGroupOverview {
    fn as_ref(&self) -> &NodeGroupOverview {
        &self.0
    }
}

#[pymethods]
impl PyNodeGroupOverview {
    #[getter]
    pub fn count(&self) -> usize {
        self.0.count
    }

    #[getter]
    pub fn attributes(&self) -> HashMap<PyMedRecordAttribute, PyAttributeOverview> {
        self.0.attributes.clone().deep_into()
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyEdgeGroupOverview(EdgeGroupOverview);

impl From<EdgeGroupOverview> for PyEdgeGroupOverview {
    fn from(value: EdgeGroupOverview) -> Self {
        Self(value)
    }
}

impl From<PyEdgeGroupOverview> for EdgeGroupOverview {
    fn from(value: PyEdgeGroupOverview) -> Self {
        value.0
    }
}

impl DeepFrom<EdgeGroupOverview> for PyEdgeGroupOverview {
    fn deep_from(value: EdgeGroupOverview) -> Self {
        value.into()
    }
}

impl DeepFrom<PyEdgeGroupOverview> for EdgeGroupOverview {
    fn deep_from(value: PyEdgeGroupOverview) -> Self {
        value.into()
    }
}

impl AsRef<EdgeGroupOverview> for PyEdgeGroupOverview {
    fn as_ref(&self) -> &EdgeGroupOverview {
        &self.0
    }
}

#[pymethods]
impl PyEdgeGroupOverview {
    #[getter]
    pub fn count(&self) -> usize {
        self.0.count
    }

    #[getter]
    pub fn attributes(&self) -> HashMap<PyMedRecordAttribute, PyAttributeOverview> {
        self.0
            .attributes
            .iter()
            .map(|(k, v)| (k.clone().into(), v.clone().into()))
            .collect()
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyGroupOverview(GroupOverview);

impl From<GroupOverview> for PyGroupOverview {
    fn from(value: GroupOverview) -> Self {
        Self(value)
    }
}

impl From<PyGroupOverview> for GroupOverview {
    fn from(value: PyGroupOverview) -> Self {
        value.0
    }
}

impl DeepFrom<GroupOverview> for PyGroupOverview {
    fn deep_from(value: GroupOverview) -> Self {
        value.into()
    }
}

impl DeepFrom<PyGroupOverview> for GroupOverview {
    fn deep_from(value: PyGroupOverview) -> Self {
        value.into()
    }
}

impl AsRef<GroupOverview> for PyGroupOverview {
    fn as_ref(&self) -> &GroupOverview {
        &self.0
    }
}

#[pymethods]
impl PyGroupOverview {
    #[getter]
    pub fn node_overview(&self) -> PyNodeGroupOverview {
        self.0.node_overview.clone().into()
    }

    #[getter]
    pub fn edge_overview(&self) -> PyEdgeGroupOverview {
        self.0.edge_overview.clone().into()
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyOverview(Overview);

impl From<Overview> for PyOverview {
    fn from(value: Overview) -> Self {
        Self(value)
    }
}

impl From<PyOverview> for Overview {
    fn from(value: PyOverview) -> Self {
        value.0
    }
}

impl AsRef<Overview> for PyOverview {
    fn as_ref(&self) -> &Overview {
        &self.0
    }
}

#[pymethods]
impl PyOverview {
    #[getter]
    pub fn ungrouped_overview(&self) -> PyGroupOverview {
        self.0.ungrouped_overview.clone().into()
    }

    #[getter]
    pub fn grouped_overviews(&self) -> HashMap<PyMedRecordAttribute, PyGroupOverview> {
        self.0.grouped_overviews.clone().deep_into()
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }
}

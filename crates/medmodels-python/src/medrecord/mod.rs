#![allow(clippy::new_without_default)]

pub mod attribute;
pub mod datatype;
pub mod errors;
pub mod overview;
pub mod querying;
pub mod schema;
pub mod traits;
pub mod value;

use crate::{
    gil_hash_map::GILHashMap,
    medrecord::overview::{PyGroupOverview, PyOverview},
};
use attribute::PyMedRecordAttribute;
use errors::PyMedRecordError;
use medmodels::{
    core::{
        errors::MedRecordError,
        medrecord::{Attributes, EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue},
    },
    utils::traits::ReadWriteOrPanic,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyFunction},
};
use pyo3_polars::PyDataFrame;
use querying::{edges::PyEdgeOperand, nodes::PyNodeOperand, PyReturnOperand};
use schema::PySchema;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use traits::DeepInto;
use value::PyMedRecordValue;

pub type PyAttributes = HashMap<PyMedRecordAttribute, PyMedRecordValue>;
pub type PyGroup = PyMedRecordAttribute;
pub type PyNodeIndex = PyMedRecordAttribute;
pub type PyEdgeIndex = EdgeIndex;
type Lut<T> = GILHashMap<usize, fn(&Bound<'_, PyAny>) -> PyResult<T>>;

#[pyclass]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyMedRecord(pub(crate) Arc<RwLock<MedRecord>>);

impl From<MedRecord> for PyMedRecord {
    fn from(value: MedRecord) -> Self {
        Self(Arc::new(RwLock::new(value)))
    }
}

impl From<PyMedRecord> for MedRecord {
    fn from(value: PyMedRecord) -> Self {
        value.0.read_or_panic().clone()
    }
}

#[pymethods]
impl PyMedRecord {
    #[new]
    pub fn new() -> Self {
        MedRecord::new().into()
    }

    pub fn _to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&(*self.0.read_or_panic()))
            .map_err(|_| MedRecordError::ConversionError("Could not serialize MedRecord".into()))
            .map_err(PyMedRecordError::from)?;

        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn _from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let medrecord: MedRecord = bincode::deserialize(data.as_bytes())
            .map_err(|_| MedRecordError::ConversionError("Could not deserialize MedRecord".into()))
            .map_err(PyMedRecordError::from)?;

        Ok(medrecord.into())
    }

    #[staticmethod]
    pub fn with_schema(schema: PySchema) -> Self {
        MedRecord::with_schema(schema.into()).into()
    }

    #[staticmethod]
    #[pyo3(signature = (nodes, edges=None))]
    pub fn from_tuples(
        nodes: Vec<(PyNodeIndex, PyAttributes)>,
        edges: Option<Vec<(PyNodeIndex, PyNodeIndex, PyAttributes)>>,
    ) -> PyResult<Self> {
        Ok(
            MedRecord::from_tuples(nodes.deep_into(), edges.deep_into(), None)
                .map_err(PyMedRecordError::from)?
                .into(),
        )
    }

    #[staticmethod]
    pub fn from_dataframes(
        nodes_dataframes: Vec<(PyDataFrame, String)>,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Self> {
        Ok(
            MedRecord::from_dataframes(nodes_dataframes, edges_dataframes, None)
                .map_err(PyMedRecordError::from)?
                .into(),
        )
    }

    #[staticmethod]
    pub fn from_nodes_dataframes(nodes_dataframes: Vec<(PyDataFrame, String)>) -> PyResult<Self> {
        Ok(MedRecord::from_nodes_dataframes(nodes_dataframes, None)
            .map_err(PyMedRecordError::from)?
            .into())
    }

    #[staticmethod]
    pub fn from_simple_example_dataset() -> Self {
        MedRecord::from_simple_example_dataset().into()
    }

    #[staticmethod]
    pub fn from_advanced_example_dataset() -> Self {
        MedRecord::from_advanced_example_dataset().into()
    }

    #[staticmethod]
    pub fn from_admissions_example_dataset() -> Self {
        MedRecord::from_admissions_example_dataset().into()
    }

    #[staticmethod]
    pub fn from_ron(path: &str) -> PyResult<Self> {
        Ok(MedRecord::from_ron(path)
            .map_err(PyMedRecordError::from)?
            .into())
    }

    pub fn to_ron(&self, path: &str) -> PyResult<()> {
        Ok(self
            .0
            .read_or_panic()
            .to_ron(path)
            .map_err(PyMedRecordError::from)?)
    }

    pub fn to_dataframes(&self, py: Python<'_>) -> PyResult<PyObject> {
        let export = self
            .0
            .read_or_panic()
            .to_dataframes()
            .map_err(PyMedRecordError::from)?;

        let outer_dict = PyDict::new(py);
        let inner_dict = PyDict::new(py);

        for (group, group_export) in export.groups {
            let group_dict = PyDict::new(py);

            let nodes_df = PyDataFrame(group_export.nodes);
            group_dict
                .set_item("nodes", nodes_df)
                .expect("Setting item must succeed");

            let edges_df = PyDataFrame(group_export.edges);
            group_dict
                .set_item("edges", edges_df)
                .expect("Setting item must succeed");

            inner_dict
                .set_item(PyMedRecordAttribute::from(group), group_dict)
                .expect("Setting item must succeed");
        }

        outer_dict
            .set_item("groups", inner_dict)
            .expect("Setting item must succeed");

        let ungrouped_dict = PyDict::new(py);

        let nodes_df = PyDataFrame(export.ungrouped.nodes);
        ungrouped_dict
            .set_item("nodes", nodes_df)
            .expect("Setting item must succeed");

        let edges_df = PyDataFrame(export.ungrouped.edges);
        ungrouped_dict
            .set_item("edges", edges_df)
            .expect("Setting item must succeed");

        outer_dict
            .set_item("ungrouped", ungrouped_dict)
            .expect("Setting item must succeed");

        Ok(outer_dict.into())
    }

    pub fn get_schema(&self) -> PySchema {
        self.0.read_or_panic().get_schema().clone().into()
    }

    pub fn set_schema(&mut self, schema: PySchema) -> PyResult<()> {
        Ok(self
            .0
            .write_or_panic()
            .set_schema(schema.into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn freeze_schema(&mut self) {
        self.0.write_or_panic().freeze_schema()
    }

    pub fn unfreeze_schema(&mut self) {
        self.0.write_or_panic().unfreeze_schema()
    }

    #[getter]
    pub fn nodes(&self) -> Vec<PyNodeIndex> {
        self.0
            .read_or_panic()
            .node_indices()
            .map(|node_index| node_index.clone().into())
            .collect()
    }

    pub fn node(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, PyAttributes>> {
        let medrecord = self.0.read_or_panic();

        node_index
            .into_iter()
            .map(|node_index| {
                let node_attributes = medrecord
                    .node_attributes(&node_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((node_index, node_attributes.clone().deep_into()))
            })
            .collect()
    }

    #[getter]
    pub fn edges(&self) -> Vec<EdgeIndex> {
        self.0.read_or_panic().edge_indices().copied().collect()
    }

    pub fn edge(&self, edge_index: Vec<EdgeIndex>) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
        let medrecord = self.0.read_or_panic();

        edge_index
            .into_iter()
            .map(|edge_index| {
                let edge_attributes = medrecord
                    .edge_attributes(&edge_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((edge_index, edge_attributes.clone().deep_into()))
            })
            .collect()
    }

    #[getter]
    pub fn groups(&self) -> Vec<PyGroup> {
        self.0
            .read_or_panic()
            .groups()
            .map(|group| group.clone().into())
            .collect()
    }

    pub fn outgoing_edges(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<EdgeIndex>>> {
        let medrecord = self.0.read_or_panic();

        node_index
            .into_iter()
            .map(|node_index| {
                let edges = medrecord
                    .outgoing_edges(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .copied()
                    .collect();

                Ok((node_index, edges))
            })
            .collect()
    }

    pub fn incoming_edges(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<EdgeIndex>>> {
        let medrecord = self.0.read_or_panic();

        node_index
            .into_iter()
            .map(|node_index| {
                let edges = medrecord
                    .incoming_edges(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .copied()
                    .collect();

                Ok((node_index, edges))
            })
            .collect()
    }

    pub fn edge_endpoints(
        &self,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<HashMap<EdgeIndex, (PyNodeIndex, PyNodeIndex)>> {
        let medrecord = self.0.read_or_panic();

        edge_index
            .into_iter()
            .map(|edge_index| {
                let edge_endpoints = medrecord
                    .edge_endpoints(&edge_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((
                    edge_index,
                    (
                        edge_endpoints.0.clone().into(),
                        edge_endpoints.1.clone().into(),
                    ),
                ))
            })
            .collect()
    }

    pub fn edges_connecting(
        &self,
        source_node_indices: Vec<PyNodeIndex>,
        target_node_indices: Vec<PyNodeIndex>,
    ) -> Vec<EdgeIndex> {
        let source_node_indices: Vec<MedRecordAttribute> = source_node_indices.deep_into();
        let target_node_indices: Vec<MedRecordAttribute> = target_node_indices.deep_into();

        self.0
            .read_or_panic()
            .edges_connecting(
                source_node_indices.iter().collect(),
                target_node_indices.iter().collect(),
            )
            .copied()
            .collect()
    }

    pub fn edges_connecting_undirected(
        &self,
        first_node_indices: Vec<PyNodeIndex>,
        second_node_indices: Vec<PyNodeIndex>,
    ) -> Vec<EdgeIndex> {
        let first_node_indices: Vec<MedRecordAttribute> = first_node_indices.deep_into();
        let second_node_indices: Vec<MedRecordAttribute> = second_node_indices.deep_into();

        self.0
            .read_or_panic()
            .edges_connecting_undirected(
                first_node_indices.iter().collect(),
                second_node_indices.iter().collect(),
            )
            .copied()
            .collect()
    }

    pub fn remove_nodes(
        &mut self,
        node_indices: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, PyAttributes>> {
        let mut medrecord = self.0.write_or_panic();

        node_indices
            .into_iter()
            .map(|node_index| {
                let attributes = medrecord
                    .remove_node(&node_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((node_index, attributes.deep_into()))
            })
            .collect()
    }

    pub fn replace_node_attributes(
        &mut self,
        node_indices: Vec<PyNodeIndex>,
        attributes: PyAttributes,
    ) -> PyResult<()> {
        let attributes: Attributes = attributes.deep_into();

        let mut medrecord = self.0.write_or_panic();

        for node_index in node_indices {
            let mut current_attributes = medrecord
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            current_attributes
                .replace_attributes(attributes.clone())
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn update_node_attribute(
        &mut self,
        node_indices: Vec<PyNodeIndex>,
        attribute: PyMedRecordAttribute,
        value: PyMedRecordValue,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();
        let value: MedRecordValue = value.into();

        let mut medrecord = self.0.write_or_panic();

        for node_index in node_indices {
            let mut node_attributes = medrecord
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            node_attributes
                .update_attribute(&attribute, value.clone())
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn remove_node_attribute(
        &mut self,
        node_indices: Vec<PyNodeIndex>,
        attribute: PyMedRecordAttribute,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();

        let mut medrecord = self.0.write_or_panic();

        for node_index in node_indices {
            let mut node_attributes = medrecord
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            node_attributes
                .remove_attribute(&attribute)
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn add_nodes(&mut self, nodes: Vec<(PyNodeIndex, PyAttributes)>) -> PyResult<()> {
        Ok(self
            .0
            .write_or_panic()
            .add_nodes(nodes.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn add_nodes_dataframes(
        &mut self,
        nodes_dataframes: Vec<(PyDataFrame, String)>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .write_or_panic()
            .add_nodes_dataframes(nodes_dataframes)
            .map_err(PyMedRecordError::from)?)
    }

    pub fn remove_edges(
        &mut self,
        edge_indices: Vec<EdgeIndex>,
    ) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
        let mut medrecord = self.0.write_or_panic();

        edge_indices
            .into_iter()
            .map(|edge_index| {
                let attributes = medrecord
                    .remove_edge(&edge_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((edge_index, attributes.deep_into()))
            })
            .collect()
    }

    pub fn replace_edge_attributes(
        &mut self,
        edge_indices: Vec<EdgeIndex>,
        attributes: PyAttributes,
    ) -> PyResult<()> {
        let attributes: Attributes = attributes.deep_into();

        let mut medrecord = self.0.write_or_panic();

        for edge_index in edge_indices {
            let mut current_attributes = medrecord
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            current_attributes
                .replace_attributes(attributes.clone())
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn update_edge_attribute(
        &mut self,
        edge_indices: Vec<EdgeIndex>,
        attribute: PyMedRecordAttribute,
        value: PyMedRecordValue,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();
        let value: MedRecordValue = value.into();

        let mut medrecord = self.0.write_or_panic();

        for edge_index in edge_indices {
            let mut edge_attributes = medrecord
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            edge_attributes
                .update_attribute(&attribute, value.clone())
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn remove_edge_attribute(
        &mut self,
        edge_indices: Vec<EdgeIndex>,
        attribute: PyMedRecordAttribute,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();

        let mut medrecord = self.0.write_or_panic();

        for edge_index in edge_indices {
            let mut edge_attributes = medrecord
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            edge_attributes
                .remove_attribute(&attribute)
                .map_err(PyMedRecordError::from)?;
        }

        Ok(())
    }

    pub fn add_edges(
        &mut self,
        relations: Vec<(PyNodeIndex, PyNodeIndex, PyAttributes)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .write_or_panic()
            .add_edges(relations.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn add_edges_dataframes(
        &mut self,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .write_or_panic()
            .add_edges_dataframes(edges_dataframes)
            .map_err(PyMedRecordError::from)?)
    }

    #[pyo3(signature = (group, node_indices_to_add=None, edge_indices_to_add=None))]
    pub fn add_group(
        &mut self,
        group: PyGroup,
        node_indices_to_add: Option<Vec<PyNodeIndex>>,
        edge_indices_to_add: Option<Vec<EdgeIndex>>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .write_or_panic()
            .add_group(
                group.into(),
                node_indices_to_add.deep_into(),
                edge_indices_to_add,
            )
            .map_err(PyMedRecordError::from)?)
    }

    pub fn remove_groups(&mut self, group: Vec<PyGroup>) -> PyResult<()> {
        let mut medrecord = self.0.write_or_panic();

        group.into_iter().try_for_each(|group| {
            medrecord
                .remove_group(&group)
                .map_err(PyMedRecordError::from)?;

            Ok(())
        })
    }

    pub fn add_nodes_to_group(
        &mut self,
        group: PyGroup,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<()> {
        let mut medrecord = self.0.write_or_panic();

        node_index.into_iter().try_for_each(|node_index| {
            Ok(medrecord
                .add_node_to_group(group.clone().into(), node_index.into())
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn add_edges_to_group(
        &mut self,
        group: PyGroup,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<()> {
        let mut medrecord = self.0.write_or_panic();

        edge_index.into_iter().try_for_each(|edge_index| {
            Ok(medrecord
                .add_edge_to_group(group.clone().into(), edge_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn remove_nodes_from_group(
        &mut self,
        group: PyGroup,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<()> {
        let mut medrecord = self.0.write_or_panic();

        node_index.into_iter().try_for_each(|node_index| {
            Ok(medrecord
                .remove_node_from_group(&group, &node_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn remove_edges_from_group(
        &mut self,
        group: PyGroup,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<()> {
        let mut medrecord = self.0.write_or_panic();

        edge_index.into_iter().try_for_each(|edge_index| {
            Ok(medrecord
                .remove_edge_from_group(&group, &edge_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn nodes_in_group(
        &self,
        group: Vec<PyGroup>,
    ) -> PyResult<HashMap<PyGroup, Vec<PyNodeIndex>>> {
        let medrecord = self.0.read_or_panic();

        group
            .into_iter()
            .map(|group| {
                let nodes_attributes = medrecord
                    .nodes_in_group(&group)
                    .map_err(PyMedRecordError::from)?
                    .map(|node_index| node_index.clone().into())
                    .collect();

                Ok((group, nodes_attributes))
            })
            .collect()
    }

    pub fn ungrouped_nodes(&self) -> Vec<PyNodeIndex> {
        self.0
            .read_or_panic()
            .ungrouped_nodes()
            .map(|node_index| node_index.clone().into())
            .collect()
    }

    pub fn edges_in_group(
        &self,
        group: Vec<PyGroup>,
    ) -> PyResult<HashMap<PyGroup, Vec<EdgeIndex>>> {
        let medrecord = self.0.read_or_panic();

        group
            .into_iter()
            .map(|group| {
                let edges = medrecord
                    .edges_in_group(&group)
                    .map_err(PyMedRecordError::from)?
                    .copied()
                    .collect();

                Ok((group, edges))
            })
            .collect()
    }

    pub fn ungrouped_edges(&self) -> Vec<EdgeIndex> {
        self.0.read_or_panic().ungrouped_edges().copied().collect()
    }

    pub fn groups_of_node(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyGroup>>> {
        let medrecord = self.0.read_or_panic();

        node_index
            .into_iter()
            .map(|node_index| {
                let groups = medrecord
                    .groups_of_node(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|node_index| node_index.clone().into())
                    .collect();

                Ok((node_index, groups))
            })
            .collect()
    }

    pub fn groups_of_edge(
        &self,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<HashMap<EdgeIndex, Vec<PyGroup>>> {
        let medrecord = self.0.read_or_panic();

        edge_index
            .into_iter()
            .map(|edge_index| {
                let groups = medrecord
                    .groups_of_edge(&edge_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|group| group.clone().into())
                    .collect();

                Ok((edge_index, groups))
            })
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.0.read_or_panic().node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.0.read_or_panic().edge_count()
    }

    pub fn group_count(&self) -> usize {
        self.0.read_or_panic().group_count()
    }

    pub fn contains_node(&self, node_index: PyNodeIndex) -> bool {
        self.0.read_or_panic().contains_node(&node_index.into())
    }

    pub fn contains_edge(&self, edge_index: EdgeIndex) -> bool {
        self.0.read_or_panic().contains_edge(&edge_index)
    }

    pub fn contains_group(&self, group: PyGroup) -> bool {
        self.0.read_or_panic().contains_group(&group.into())
    }

    pub fn neighbors(
        &self,
        node_indices: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyNodeIndex>>> {
        let medrecord = self.0.read_or_panic();

        node_indices
            .into_iter()
            .map(|node_index| {
                let neighbors = medrecord
                    .neighbors_outgoing(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|neighbor| neighbor.clone().into())
                    .collect();

                Ok((node_index, neighbors))
            })
            .collect()
    }

    pub fn neighbors_undirected(
        &self,
        node_indices: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyNodeIndex>>> {
        let medrecord = self.0.read_or_panic();

        node_indices
            .into_iter()
            .map(|node_index| {
                let neighbors = medrecord
                    .neighbors_undirected(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|neighbor| neighbor.clone().into())
                    .collect();

                Ok((node_index, neighbors))
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.0.write_or_panic().clear();
    }

    pub fn query_nodes<'py>(&self, query: &Bound<'py, PyFunction>) -> PyResult<Bound<'py, PyAny>> {
        let medrecord = self.0.read_or_panic();

        let py = query.py();

        let return_values = medrecord
            .query_nodes(|nodes| {
                let result = query
                    .call1((PyNodeOperand::from(nodes.clone()),))
                    .expect("Call must succeed");

                result
                    .extract::<PyReturnOperand>()
                    .expect("Extraction must succeed")
            })
            .evaluate()
            .map_err(PyMedRecordError::from)?;

        return_values.into_pyobject(py)
    }

    pub fn query_edges<'py>(&self, query: &Bound<'py, PyFunction>) -> PyResult<Bound<'py, PyAny>> {
        let medrecord = self.0.read_or_panic();

        let py = query.py();

        let return_values = medrecord
            .query_edges(|edges| {
                let result = query
                    .call1((PyEdgeOperand::from(edges.clone()),))
                    .expect("Call must succeed");

                result
                    .extract::<PyReturnOperand>()
                    .expect("Extraction must succeed")
            })
            .evaluate()
            .map_err(PyMedRecordError::from)?;

        return_values.into_pyobject(py)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Self {
        self.0.read_or_panic().clone().into()
    }

    pub fn overview(&self, truncate_details: Option<usize>) -> PyResult<PyOverview> {
        Ok(self
            .0
            .read_or_panic()
            .overview(truncate_details)
            .map_err(PyMedRecordError::from)?
            .into())
    }

    pub fn group_overview(
        &self,
        group: PyGroup,
        truncate_details: Option<usize>,
    ) -> PyResult<PyGroupOverview> {
        Ok(self
            .0
            .read_or_panic()
            .group_overview(&group.into(), truncate_details)
            .map_err(PyMedRecordError::from)?
            .into())
    }
}

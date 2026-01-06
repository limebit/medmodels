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
use medmodels::core::{
    errors::MedRecordError,
    medrecord::{Attributes, EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue},
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyFunction},
};
use pyo3_polars::PyDataFrame;
use querying::{edges::PyEdgeOperand, nodes::PyNodeOperand, PyReturnOperand, PyReturnValue};
use schema::PySchema;
use std::collections::HashMap;
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
pub struct PyMedRecord(MedRecord);

impl From<MedRecord> for PyMedRecord {
    fn from(value: MedRecord) -> Self {
        Self(value)
    }
}

impl From<PyMedRecord> for MedRecord {
    fn from(value: PyMedRecord) -> Self {
        value.0
    }
}

impl AsRef<MedRecord> for PyMedRecord {
    fn as_ref(&self) -> &MedRecord {
        &self.0
    }
}

#[pymethods]
impl PyMedRecord {
    #[new]
    pub fn new() -> Self {
        Self(MedRecord::new())
    }

    pub fn _to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self.0)
            .map_err(|_| MedRecordError::ConversionError("Could not serialize MedRecord".into()))
            .map_err(PyMedRecordError::from)?;

        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn _from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let medrecord = bincode::deserialize(data.as_bytes())
            .map_err(|_| MedRecordError::ConversionError("Could not deserialize MedRecord".into()))
            .map_err(PyMedRecordError::from)?;

        Ok(Self(medrecord))
    }

    #[staticmethod]
    pub fn with_schema(schema: PySchema) -> Self {
        Self(MedRecord::with_schema(schema.into()))
    }

    #[staticmethod]
    #[pyo3(signature = (nodes, edges=None))]
    pub fn from_tuples(
        nodes: Vec<(PyNodeIndex, PyAttributes)>,
        edges: Option<Vec<(PyNodeIndex, PyNodeIndex, PyAttributes)>>,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_tuples(nodes.deep_into(), edges.deep_into(), None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    pub fn from_dataframes(
        nodes_dataframes: Vec<(PyDataFrame, String)>,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_dataframes(nodes_dataframes, edges_dataframes, None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    pub fn from_nodes_dataframes(nodes_dataframes: Vec<(PyDataFrame, String)>) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_nodes_dataframes(nodes_dataframes, None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    pub fn from_simple_example_dataset() -> Self {
        Self(MedRecord::from_simple_example_dataset())
    }

    #[staticmethod]
    pub fn from_advanced_example_dataset() -> Self {
        Self(MedRecord::from_advanced_example_dataset())
    }

    #[staticmethod]
    pub fn from_admissions_example_dataset() -> Self {
        Self(MedRecord::from_admissions_example_dataset())
    }

    #[staticmethod]
    pub fn from_ron(path: &str) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_ron(path).map_err(PyMedRecordError::from)?,
        ))
    }

    pub fn to_ron(&self, path: &str) -> PyResult<()> {
        Ok(self.0.to_ron(path).map_err(PyMedRecordError::from)?)
    }

    pub fn to_dataframes(&self, py: Python<'_>) -> PyResult<PyObject> {
        let export = self.0.to_dataframes().map_err(PyMedRecordError::from)?;

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
        self.0.get_schema().clone().into()
    }

    pub fn set_schema(&mut self, schema: PySchema) -> PyResult<()> {
        Ok(self
            .0
            .set_schema(schema.into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn freeze_schema(&mut self) {
        self.0.freeze_schema()
    }

    pub fn unfreeze_schema(&mut self) {
        self.0.unfreeze_schema()
    }

    #[getter]
    pub fn nodes(&self) -> Vec<PyNodeIndex> {
        self.0
            .node_indices()
            .map(|node_index| node_index.clone().into())
            .collect()
    }

    pub fn node(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, PyAttributes>> {
        node_index
            .into_iter()
            .map(|node_index| {
                let node_attributes = self
                    .0
                    .node_attributes(&node_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((node_index, node_attributes.clone().deep_into()))
            })
            .collect()
    }

    #[getter]
    pub fn edges(&self) -> Vec<EdgeIndex> {
        self.0.edge_indices().copied().collect()
    }

    pub fn edge(&self, edge_index: Vec<EdgeIndex>) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
        edge_index
            .into_iter()
            .map(|edge_index| {
                let edge_attributes = self
                    .0
                    .edge_attributes(&edge_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((edge_index, edge_attributes.clone().deep_into()))
            })
            .collect()
    }

    #[getter]
    pub fn groups(&self) -> Vec<PyGroup> {
        self.0.groups().map(|group| group.clone().into()).collect()
    }

    pub fn outgoing_edges(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<EdgeIndex>>> {
        node_index
            .into_iter()
            .map(|node_index| {
                let edges = self
                    .0
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
        node_index
            .into_iter()
            .map(|node_index| {
                let edges = self
                    .0
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
        edge_index
            .into_iter()
            .map(|edge_index| {
                let edge_endpoints = self
                    .0
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
        node_indices
            .into_iter()
            .map(|node_index| {
                let attributes = self
                    .0
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

        for node_index in node_indices {
            let mut current_attributes = self
                .0
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

        for node_index in node_indices {
            let mut node_attributes = self
                .0
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

        for node_index in node_indices {
            let mut node_attributes = self
                .0
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
            .add_nodes(nodes.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn add_nodes_dataframes(
        &mut self,
        nodes_dataframes: Vec<(PyDataFrame, String)>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_nodes_dataframes(nodes_dataframes)
            .map_err(PyMedRecordError::from)?)
    }

    pub fn remove_edges(
        &mut self,
        edge_indices: Vec<EdgeIndex>,
    ) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
        edge_indices
            .into_iter()
            .map(|edge_index| {
                let attributes = self
                    .0
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

        for edge_index in edge_indices {
            let mut current_attributes = self
                .0
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

        for edge_index in edge_indices {
            let mut edge_attributes = self
                .0
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

        for edge_index in edge_indices {
            let mut edge_attributes = self
                .0
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
            .add_edges(relations.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn add_edges_dataframes(
        &mut self,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
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
            .add_group(
                group.into(),
                node_indices_to_add.deep_into(),
                edge_indices_to_add,
            )
            .map_err(PyMedRecordError::from)?)
    }

    pub fn remove_groups(&mut self, group: Vec<PyGroup>) -> PyResult<()> {
        group.into_iter().try_for_each(|group| {
            self.0
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
        node_index.into_iter().try_for_each(|node_index| {
            Ok(self
                .0
                .add_node_to_group(group.clone().into(), node_index.into())
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn add_edges_to_group(
        &mut self,
        group: PyGroup,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<()> {
        edge_index.into_iter().try_for_each(|edge_index| {
            Ok(self
                .0
                .add_edge_to_group(group.clone().into(), edge_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn remove_nodes_from_group(
        &mut self,
        group: PyGroup,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<()> {
        node_index.into_iter().try_for_each(|node_index| {
            Ok(self
                .0
                .remove_node_from_group(&group, &node_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn remove_edges_from_group(
        &mut self,
        group: PyGroup,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<()> {
        edge_index.into_iter().try_for_each(|edge_index| {
            Ok(self
                .0
                .remove_edge_from_group(&group, &edge_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    pub fn nodes_in_group(
        &self,
        group: Vec<PyGroup>,
    ) -> PyResult<HashMap<PyGroup, Vec<PyNodeIndex>>> {
        group
            .into_iter()
            .map(|group| {
                let nodes_attributes = self
                    .0
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
            .ungrouped_nodes()
            .map(|node_index| node_index.clone().into())
            .collect()
    }

    pub fn edges_in_group(
        &self,
        group: Vec<PyGroup>,
    ) -> PyResult<HashMap<PyGroup, Vec<EdgeIndex>>> {
        group
            .into_iter()
            .map(|group| {
                let edges = self
                    .0
                    .edges_in_group(&group)
                    .map_err(PyMedRecordError::from)?
                    .copied()
                    .collect();

                Ok((group, edges))
            })
            .collect()
    }

    pub fn ungrouped_edges(&self) -> Vec<EdgeIndex> {
        self.0.ungrouped_edges().copied().collect()
    }

    pub fn groups_of_node(
        &self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyGroup>>> {
        node_index
            .into_iter()
            .map(|node_index| {
                let groups = self
                    .0
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
        edge_index
            .into_iter()
            .map(|edge_index| {
                let groups = self
                    .0
                    .groups_of_edge(&edge_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|group| group.clone().into())
                    .collect();

                Ok((edge_index, groups))
            })
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    pub fn group_count(&self) -> usize {
        self.0.group_count()
    }

    pub fn contains_node(&self, node_index: PyNodeIndex) -> bool {
        self.0.contains_node(&node_index.into())
    }

    pub fn contains_edge(&self, edge_index: EdgeIndex) -> bool {
        self.0.contains_edge(&edge_index)
    }

    pub fn contains_group(&self, group: PyGroup) -> bool {
        self.0.contains_group(&group.into())
    }

    pub fn neighbors(
        &self,
        node_indices: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyNodeIndex>>> {
        node_indices
            .into_iter()
            .map(|node_index| {
                let neighbors = self
                    .0
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
        node_indices
            .into_iter()
            .map(|node_index| {
                let neighbors = self
                    .0
                    .neighbors_undirected(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|neighbor| neighbor.clone().into())
                    .collect();

                Ok((node_index, neighbors))
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn query_nodes(&self, query: &Bound<'_, PyFunction>) -> PyResult<PyReturnValue<'_>> {
        Ok(self
            .0
            .query_nodes(|nodes| {
                let result = query
                    .call1((PyNodeOperand::from(nodes.clone()),))
                    .expect("Call must succeed");

                result
                    .extract::<PyReturnOperand>()
                    .expect("Extraction must succeed")
            })
            .evaluate()
            .map_err(PyMedRecordError::from)?)
    }

    pub fn query_edges(&self, query: &Bound<'_, PyFunction>) -> PyResult<PyReturnValue<'_>> {
        Ok(self
            .0
            .query_edges(|edges| {
                let result = query
                    .call1((PyEdgeOperand::from(edges.clone()),))
                    .expect("Call must succeed");

                result
                    .extract::<PyReturnOperand>()
                    .expect("Extraction must succeed")
            })
            .evaluate()
            .map_err(PyMedRecordError::from)?)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    pub fn overview(&self, truncate_details: Option<usize>) -> PyResult<PyOverview> {
        Ok(self
            .0
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
            .group_overview(&group.into(), truncate_details)
            .map_err(PyMedRecordError::from)?
            .into())
    }
}

mod attribute;
pub mod datatype;
mod errors;
pub mod schema;
mod traits;
mod value;

use crate::gil_hash_map::GILHashMap;
use attribute::PyMedRecordAttribute;
use errors::PyMedRecordError;
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{Attributes, EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue},
};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use schema::PySchema;
use std::collections::HashMap;
use traits::DeepInto;
use value::PyMedRecordValue;

type PyAttributes = HashMap<PyMedRecordAttribute, PyMedRecordValue>;
type PyGroup = PyMedRecordAttribute;
type PyNodeIndex = PyMedRecordAttribute;
type Lut<T> = GILHashMap<usize, fn(&Bound<'_, PyAny>) -> PyResult<T>>;

#[pyclass]
#[repr(transparent)]
pub struct PyMedRecord(MedRecord);

#[pymethods]
impl PyMedRecord {
    #[new]
    fn new() -> Self {
        Self(MedRecord::new())
    }

    #[staticmethod]
    fn with_schema(schema: PySchema) -> Self {
        Self(MedRecord::with_schema(schema.into()))
    }

    #[staticmethod]
    fn from_tuples(
        nodes: Vec<(PyNodeIndex, PyAttributes)>,
        edges: Option<Vec<(PyNodeIndex, PyNodeIndex, PyAttributes)>>,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_tuples(nodes.deep_into(), edges.deep_into(), None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    fn from_dataframes(
        nodes_dataframes: Vec<(PyDataFrame, String)>,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_dataframes(nodes_dataframes, edges_dataframes, None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    fn from_nodes_dataframes(nodes_dataframes: Vec<(PyDataFrame, String)>) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_nodes_dataframes(nodes_dataframes, None)
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    fn from_example_dataset() -> Self {
        Self(MedRecord::from_example_dataset())
    }

    #[staticmethod]
    fn from_ron(path: &str) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_ron(path).map_err(PyMedRecordError::from)?,
        ))
    }

    fn to_ron(&self, path: &str) -> PyResult<()> {
        Ok(self.0.to_ron(path).map_err(PyMedRecordError::from)?)
    }

    fn update_schema(&mut self, schema: PySchema) -> PyResult<()> {
        Ok(self
            .0
            .update_schema(schema.into())
            .map_err(PyMedRecordError::from)?)
    }

    #[getter]
    fn schema(&self) -> PySchema {
        self.0.get_schema().clone().into()
    }

    #[getter]
    fn nodes(&self) -> Vec<PyNodeIndex> {
        self.0
            .node_indices()
            .map(|node_index| node_index.clone().into())
            .collect()
    }

    fn node(&self, node_index: Vec<PyNodeIndex>) -> PyResult<HashMap<PyNodeIndex, PyAttributes>> {
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
    fn edges(&self) -> Vec<EdgeIndex> {
        self.0.edge_indices().copied().collect()
    }

    fn edge(&self, edge_index: Vec<EdgeIndex>) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
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
    fn groups(&self) -> Vec<PyGroup> {
        self.0.groups().map(|group| group.clone().into()).collect()
    }

    fn outgoing_edges(
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

    fn incoming_edges(
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

    fn edge_endpoints(
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

    fn edges_connecting(
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

    fn edges_connecting_undirected(
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

    fn add_node(&mut self, node_index: PyNodeIndex, attributes: PyAttributes) -> PyResult<()> {
        Ok(self
            .0
            .add_node(node_index.into(), attributes.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    fn remove_node(
        &mut self,
        node_index: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, PyAttributes>> {
        node_index
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

    fn replace_node_attributes(
        &mut self,
        node_index: Vec<PyNodeIndex>,
        attributes: PyAttributes,
    ) -> PyResult<()> {
        let attributes: Attributes = attributes.deep_into();

        for node_index in node_index {
            let current_attributes = self
                .0
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            current_attributes.clone_from(&attributes);
        }

        Ok(())
    }

    fn update_node_attribute(
        &mut self,
        node_index: Vec<PyNodeIndex>,
        attribute: PyMedRecordAttribute,
        value: PyMedRecordValue,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();
        let value: MedRecordValue = value.into();

        for node_index in node_index {
            let node_attributes = self
                .0
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            node_attributes
                .entry(attribute.clone())
                .or_default()
                .clone_from(&value)
        }

        Ok(())
    }

    fn remove_node_attribute(
        &mut self,
        node_index: Vec<PyNodeIndex>,
        attribute: PyMedRecordAttribute,
    ) -> PyResult<()> {
        let attribute: MedRecordAttribute = attribute.into();

        for node_index in node_index {
            let node_attributes = self
                .0
                .node_attributes_mut(&node_index)
                .map_err(PyMedRecordError::from)?;

            node_attributes
                .remove(&attribute)
                .ok_or(PyMedRecordError::from(MedRecordError::KeyError(format!(
                    "Cannot find attribute {} in node {}",
                    attribute, *node_index
                ))))?;
        }

        Ok(())
    }

    fn add_nodes(&mut self, nodes: Vec<(PyNodeIndex, PyAttributes)>) -> PyResult<()> {
        Ok(self
            .0
            .add_nodes(nodes.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    fn add_nodes_dataframes(
        &mut self,
        nodes_dataframes: Vec<(PyDataFrame, String)>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_nodes_dataframes(nodes_dataframes)
            .map_err(PyMedRecordError::from)?)
    }

    fn add_edge(
        &mut self,
        source_node_index: PyNodeIndex,
        target_node_index: PyNodeIndex,
        attributes: PyAttributes,
    ) -> PyResult<EdgeIndex> {
        Ok(self
            .0
            .add_edge(
                source_node_index.into(),
                target_node_index.into(),
                attributes.deep_into(),
            )
            .map_err(PyMedRecordError::from)?)
    }

    fn remove_edge(
        &mut self,
        edge_index: Vec<EdgeIndex>,
    ) -> PyResult<HashMap<EdgeIndex, PyAttributes>> {
        edge_index
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

    fn replace_edge_attributes(
        &mut self,
        edge_index: Vec<EdgeIndex>,
        attributes: PyAttributes,
    ) -> PyResult<()> {
        let attributes: Attributes = attributes.deep_into();

        for edge_index in edge_index {
            let current_attributes = self
                .0
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            current_attributes.clone_from(&attributes);
        }

        Ok(())
    }

    fn update_edge_attribute(
        &mut self,
        edge_index: Vec<EdgeIndex>,
        attribute: PyMedRecordAttribute,
        value: PyMedRecordValue,
    ) -> PyResult<()> {
        for edge_index in edge_index {
            let edge_attributes = self
                .0
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            edge_attributes
                .entry((*attribute).clone())
                .or_default()
                .clone_from(&value);
        }

        Ok(())
    }

    fn remove_edge_attribute(
        &mut self,
        edge_index: Vec<EdgeIndex>,
        attribute: PyMedRecordAttribute,
    ) -> PyResult<()> {
        for edge_index in edge_index {
            let edge_attributes = self
                .0
                .edge_attributes_mut(&edge_index)
                .map_err(PyMedRecordError::from)?;

            edge_attributes
                .remove(&attribute)
                .ok_or(PyMedRecordError::from(MedRecordError::KeyError(format!(
                    "Cannot find attribute {} in edge {}",
                    *attribute, edge_index
                ))))?;
        }

        Ok(())
    }

    fn add_edges(
        &mut self,
        relations: Vec<(PyNodeIndex, PyNodeIndex, PyAttributes)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .add_edges(relations.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    fn add_edges_dataframes(
        &mut self,
        edges_dataframes: Vec<(PyDataFrame, String, String)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .add_edges_dataframes(edges_dataframes)
            .map_err(PyMedRecordError::from)?)
    }

    fn add_group(
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

    fn remove_group(&mut self, group: Vec<PyGroup>) -> PyResult<()> {
        group.into_iter().try_for_each(|group| {
            self.0
                .remove_group(&group)
                .map_err(PyMedRecordError::from)?;

            Ok(())
        })
    }

    fn add_node_to_group(&mut self, group: PyGroup, node_index: Vec<PyNodeIndex>) -> PyResult<()> {
        node_index.into_iter().try_for_each(|node_index| {
            Ok(self
                .0
                .add_node_to_group(group.clone().into(), node_index.into())
                .map_err(PyMedRecordError::from)?)
        })
    }

    fn add_edge_to_group(&mut self, group: PyGroup, edge_index: Vec<EdgeIndex>) -> PyResult<()> {
        edge_index.into_iter().try_for_each(|edge_index| {
            Ok(self
                .0
                .add_edge_to_group(group.clone().into(), edge_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    fn remove_node_from_group(
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

    fn remove_edge_from_group(
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

    fn nodes_in_group(&self, group: Vec<PyGroup>) -> PyResult<HashMap<PyGroup, Vec<PyNodeIndex>>> {
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

    fn edges_in_group(&self, group: Vec<PyGroup>) -> PyResult<HashMap<PyGroup, Vec<EdgeIndex>>> {
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

    fn groups_of_node(
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

    fn groups_of_edge(
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

    fn node_count(&self) -> usize {
        self.0.node_count()
    }

    fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    fn group_count(&self) -> usize {
        self.0.group_count()
    }

    fn contains_node(&self, node_index: PyNodeIndex) -> bool {
        self.0.contains_node(&node_index.into())
    }

    fn contains_edge(&self, edge_index: EdgeIndex) -> bool {
        self.0.contains_edge(&edge_index)
    }

    fn contains_group(&self, group: PyGroup) -> bool {
        self.0.contains_group(&group.into())
    }

    fn neighbors(
        &self,
        node_indices: Vec<PyNodeIndex>,
    ) -> PyResult<HashMap<PyNodeIndex, Vec<PyNodeIndex>>> {
        node_indices
            .into_iter()
            .map(|node_index| {
                let neighbors = self
                    .0
                    .neighbors(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|neighbor| neighbor.clone().into())
                    .collect();

                Ok((node_index, neighbors))
            })
            .collect()
    }

    fn neighbors_undirected(
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

    fn clear(&mut self) {
        self.0.clear();
    }
}

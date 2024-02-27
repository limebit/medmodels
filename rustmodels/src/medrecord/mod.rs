mod conversion;
mod errors;

use conversion::{DeepInto, PyMedRecordAttribute, PyMedRecordValue};
use errors::PyMedRecordError;
use medmodels_core::medrecord::{EdgeIndex, MedRecord};
use pyo3::{prelude::*, types::PyTuple};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

type Dictionary = HashMap<PyMedRecordAttribute, PyMedRecordValue>;
type PyGroup = PyMedRecordAttribute;
type PyNodeIndex = PyMedRecordAttribute;

#[pyclass]
pub struct PyMedRecord(pub MedRecord);

#[pymethods]
impl PyMedRecord {
    #[new]
    fn new() -> Self {
        Self(MedRecord::new())
    }

    #[staticmethod]
    fn from_tuples(
        nodes: Vec<(PyNodeIndex, Dictionary)>,
        edges: Option<Vec<(PyNodeIndex, PyNodeIndex, Dictionary)>>,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_tuples(nodes.deep_into(), edges.deep_into())
                .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    fn from_dataframes(
        nodes_dataframe: PyDataFrame,
        nodes_index_column_name: &str,
        edges_dataframe: PyDataFrame,
        edges_from_index_column_name: &str,
        edges_to_index_column_name: &str,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_dataframes(
                nodes_dataframe.into(),
                nodes_index_column_name,
                edges_dataframe.into(),
                edges_from_index_column_name,
                edges_to_index_column_name,
            )
            .map_err(PyMedRecordError::from)?,
        ))
    }

    #[staticmethod]
    fn from_nodes_dataframe(
        nodes_dataframe: PyDataFrame,
        nodes_index_column_name: &str,
    ) -> PyResult<Self> {
        Ok(Self(
            MedRecord::from_nodes_dataframe(nodes_dataframe.into(), nodes_index_column_name)
                .map_err(PyMedRecordError::from)?,
        ))
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

    #[getter]
    fn nodes(&self) -> Vec<PyNodeIndex> {
        self.0
            .node_indices()
            .map(|node_index| node_index.to_owned().into())
            .collect()
    }

    #[pyo3(signature = (*node_index))]
    fn node(&self, node_index: &PyTuple) -> PyResult<HashMap<PyNodeIndex, Dictionary>> {
        node_index
            .iter()
            .map(|node_index| {
                let node_index = node_index.extract::<PyNodeIndex>()?.into();

                let node_attributes = self
                    .0
                    .node_attributes(&node_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((node_index.into(), node_attributes.to_owned().deep_into()))
            })
            .collect()
    }

    #[getter]
    fn edges(&self) -> Vec<EdgeIndex> {
        self.0
            .edge_indices()
            .map(|edge_index| edge_index.to_owned())
            .collect()
    }

    #[pyo3(signature = (*edge_index))]
    fn edge(&self, edge_index: &PyTuple) -> PyResult<HashMap<EdgeIndex, Dictionary>> {
        edge_index
            .iter()
            .map(|edge_index| {
                let edge_index = edge_index.extract::<EdgeIndex>()?;

                let edge_attributes = self
                    .0
                    .edge_attributes(&edge_index)
                    .map_err(PyMedRecordError::from)?;

                Ok((edge_index, edge_attributes.to_owned().deep_into()))
            })
            .collect()
    }

    fn edges_connecting(
        &self,
        source_node_index: PyNodeIndex,
        target_node_index: PyNodeIndex,
    ) -> Vec<EdgeIndex> {
        self.0
            .edges_connecting(&source_node_index.into(), &target_node_index.into())
            .map(|edge_index| edge_index.to_owned())
            .collect()
    }

    #[getter]
    fn groups(&self) -> Vec<PyGroup> {
        self.0
            .groups()
            .map(|group| group.to_owned().into())
            .collect()
    }

    #[pyo3(signature = (*group))]
    fn group(&self, group: &PyTuple) -> PyResult<HashMap<PyGroup, Vec<PyNodeIndex>>> {
        group
            .into_iter()
            .map(|group| {
                let group_id = group.extract::<PyGroup>()?.into();

                let nodes_attributes = self
                    .0
                    .nodes_in_group(&group_id)
                    .map_err(PyMedRecordError::from)?
                    .map(|node_index| node_index.to_owned().into())
                    .collect();

                Ok((group_id.into(), nodes_attributes))
            })
            .collect()
    }

    fn add_node(&mut self, node_index: PyNodeIndex, attributes: Dictionary) {
        self.0.add_node(node_index.into(), attributes.deep_into())
    }

    fn add_nodes(&mut self, nodes: Vec<(PyNodeIndex, Dictionary)>) {
        self.0.add_nodes(nodes.deep_into())
    }

    fn add_nodes_dataframe(
        &mut self,
        nodes_dataframe: PyDataFrame,
        index_column_name: &str,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_nodes_dataframe(nodes_dataframe.into(), index_column_name)
            .map_err(PyMedRecordError::from)?)
    }

    fn add_edge(
        &mut self,
        source_node_index: PyNodeIndex,
        target_node_index: PyNodeIndex,
        attributes: Dictionary,
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

    fn add_edges(
        &mut self,
        relations: Vec<(PyNodeIndex, PyNodeIndex, Dictionary)>,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .add_edges(relations.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    fn add_edges_dataframe(
        &mut self,
        edges_dataframe: PyDataFrame,
        source_index_column_name: &str,
        target_index_column_name: &str,
    ) -> PyResult<Vec<EdgeIndex>> {
        Ok(self
            .0
            .add_edges_dataframe(
                edges_dataframe.into(),
                source_index_column_name,
                target_index_column_name,
            )
            .map_err(PyMedRecordError::from)?)
    }

    fn add_group(
        &mut self,
        group: PyGroup,
        node_indices_to_add: Option<Vec<PyNodeIndex>>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_group(group.into(), node_indices_to_add.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    fn remove_group(&mut self, group: PyGroup) -> PyResult<()> {
        Ok(self
            .0
            .remove_group(&group.into())
            .map_err(PyMedRecordError::from)?)
    }

    fn remove_from_group(&mut self, group: PyGroup, node_index: PyNodeIndex) -> PyResult<()> {
        Ok(self
            .0
            .remove_from_group(&group.into(), &node_index.into())
            .map_err(PyMedRecordError::from)?)
    }

    #[pyo3(signature = (group, *node_index))]
    fn add_to_group(&mut self, group: PyGroup, node_index: &PyTuple) -> PyResult<()> {
        let group = group.into();

        node_index.into_iter().try_for_each(|node_index| {
            let node_index = node_index.extract::<PyNodeIndex>()?.into();

            Ok(self
                .0
                .add_to_group(&group, node_index)
                .map_err(PyMedRecordError::from)?)
        })
    }

    #[pyo3(signature = (*node_index))]
    fn neighbors(&self, node_index: &PyTuple) -> PyResult<HashMap<PyNodeIndex, Vec<PyNodeIndex>>> {
        node_index
            .into_iter()
            .map(|node_index| {
                let node_index = node_index.extract::<PyNodeIndex>()?.into();

                let neighbors = self
                    .0
                    .neighbors(&node_index)
                    .map_err(PyMedRecordError::from)?
                    .map(|neighbor| neighbor.to_owned().into())
                    .collect();

                Ok((node_index.into(), neighbors))
            })
            .collect()
    }
}

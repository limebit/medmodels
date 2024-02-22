mod conversion;
mod errors;

use self::{
    conversion::{DeepInto, PyMedRecordValue},
    errors::PyMedRecordError,
};
use medmodels_core::medrecord::MedRecord;
use pyo3::{prelude::*, types::PyTuple};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

type Dictionary = HashMap<String, PyMedRecordValue>;
type Group = String;

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
        nodes: Vec<(String, Dictionary)>,
        edges: Option<Vec<(String, String, Dictionary)>>,
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
    fn nodes(&self) -> Vec<&String> {
        self.0.nodes()
    }

    #[pyo3(signature = (*node_id))]
    fn node(&self, node_id: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
        Ok(self
            .0
            .node(
                node_id
                    .iter()
                    .map(|item| item.extract::<String>())
                    .collect::<Result<_, _>>()?,
            )
            .map(|nodes| {
                nodes
                    .into_iter()
                    .map(|(id, weight)| (id, weight.to_owned().deep_into()))
                    .collect()
            })
            .map_err(PyMedRecordError::from)?)
    }

    #[getter]
    fn edges(&self) -> PyResult<Vec<(&String, &String)>> {
        Ok(self.0.edges().map_err(PyMedRecordError::from)?)
    }

    fn edges_between(&self, start_node_id: &str, end_node_id: &str) -> PyResult<Vec<Dictionary>> {
        Ok(self
            .0
            .edges_between(start_node_id, end_node_id)
            .map(|edges| {
                edges
                    .into_iter()
                    .map(|weight| weight.to_owned().deep_into())
                    .collect()
            })
            .map_err(PyMedRecordError::from)?)
    }

    #[getter]
    fn groups(&self) -> Vec<&String> {
        self.0.groups()
    }

    #[pyo3(signature = (*group))]
    fn group(&self, group: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
        Ok(self
            .0
            .group(
                group
                    .iter()
                    .map(|item| item.extract::<String>())
                    .collect::<Result<_, _>>()?,
            )
            .map(|nodes| {
                nodes
                    .into_iter()
                    .map(|(id, weight)| (id.to_owned(), weight.to_owned().deep_into()))
                    .collect()
            })
            .map_err(PyMedRecordError::from)?)
    }

    fn add_nodes(&mut self, nodes: Vec<(String, Dictionary)>) {
        self.0.add_nodes(nodes.deep_into())
    }

    pub fn add_nodes_dataframe(
        &mut self,
        nodes_dataframe: PyDataFrame,
        index_column_name: &str,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_nodes_dataframe(nodes_dataframe.into(), index_column_name)
            .map_err(PyMedRecordError::from)?)
    }

    fn add_edges(&mut self, relations: Vec<(String, String, Dictionary)>) -> PyResult<()> {
        Ok(self
            .0
            .add_edges(relations.deep_into())
            .map_err(PyMedRecordError::from)?)
    }

    pub fn add_edges_dataframe(
        &mut self,
        edges_dataframe: PyDataFrame,
        from_index_column_name: &str,
        to_index_column_name: &str,
    ) -> PyResult<()> {
        Ok(self
            .0
            .add_edges_dataframe(
                edges_dataframe.into(),
                from_index_column_name,
                to_index_column_name,
            )
            .map_err(PyMedRecordError::from)?)
    }

    fn add_group(&mut self, group: Group, node_ids_to_add: Option<Vec<String>>) -> PyResult<()> {
        Ok(self
            .0
            .add_group(group, node_ids_to_add)
            .map_err(PyMedRecordError::from)?)
    }

    fn remove_group(&mut self, group: &str) -> PyResult<()> {
        Ok(self.0.remove_group(group).map_err(PyMedRecordError::from)?)
    }

    fn remove_from_group(&mut self, group: Group, node_id: &str) -> PyResult<()> {
        Ok(self
            .0
            .remove_from_group(group, node_id)
            .map_err(PyMedRecordError::from)?)
    }

    fn add_to_group(&mut self, group: Group, node_id: String) -> PyResult<()> {
        Ok(self
            .0
            .add_to_group(group, node_id)
            .map_err(PyMedRecordError::from)?)
    }

    #[pyo3(signature = (*node_id))]
    fn neighbors(&self, node_id: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
        Ok(self
            .0
            .neighbors(
                node_id
                    .iter()
                    .map(|item| item.extract::<String>())
                    .collect::<Result<_, _>>()?,
            )
            .map(|neighbors| {
                neighbors
                    .into_iter()
                    .map(|(id, weight)| (id, weight.to_owned().deep_into()))
                    .collect()
            })
            .map_err(PyMedRecordError::from)?)
    }
}

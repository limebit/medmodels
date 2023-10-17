use std::collections::HashMap;

use crate::index_mapping::IndexMapping;
use crate::py_any_value::PyAnyValue;
use petgraph::{
    data::{Element, FromElements},
    stable_graph::{NodeIndex, StableGraph},
    Directed,
};
use polars::prelude::DataFrame;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError},
    prelude::*,
    types::PyTuple,
};
use pyo3_polars::PyDataFrame;

type Dictionary = HashMap<String, PyObject>;

#[pyclass]
pub struct Medrecord {
    graph: StableGraph<Dictionary, Dictionary, Directed>,
    index_mapping: IndexMapping,
}

#[pymethods]
impl Medrecord {
    #[new]
    fn new() -> Medrecord {
        Medrecord {
            graph: StableGraph::default(),
            index_mapping: IndexMapping::new(),
        }
    }

    #[staticmethod]
    fn from_nodes_and_edges(
        nodes: Vec<(String, Dictionary)>,
        edges: Vec<(String, String, Dictionary)>,
    ) -> PyResult<Medrecord> {
        let mut index_mapping = IndexMapping::new();

        let mut elements = Vec::<Element<Dictionary, Dictionary>>::new();

        for (index, (id, weight)) in nodes.iter().enumerate() {
            index_mapping.insert_custom_index_to_node_index(id.to_owned(), NodeIndex::new(index));

            elements.push(Element::Node {
                weight: weight.to_owned(),
            });
        }

        for (id_node_1, id_node_2, weight) in edges.iter() {
            let node_index_node_1 =
                index_mapping
                    .get_node_index(id_node_1)
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        id_node_1
                    )))?;

            let node_index_node_2 =
                index_mapping
                    .get_node_index(id_node_2)
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        id_node_2
                    )))?;

            elements.push(Element::Edge {
                source: node_index_node_1.index(),
                target: node_index_node_2.index(),
                weight: weight.to_owned(),
            });
        }

        Ok(Medrecord {
            graph: StableGraph::from_elements(elements),
            index_mapping: index_mapping.to_owned(),
        })
    }

    #[staticmethod]
    fn from_dataframes(
        nodes_dataframe: PyDataFrame,
        edges_dataframe: PyDataFrame,
        node_index_column_name: String,
        node_attribute_column_names: Vec<String>,
        edge_from_index_column_name: String,
        edge_to_index_column_name: String,
        edge_attribute_column_names: Vec<String>,
    ) -> PyResult<Medrecord> {
        let mut elements = Vec::<Element<Dictionary, Dictionary>>::new();
        let mut index_mapping = IndexMapping::new();

        let nodes: DataFrame = nodes_dataframe.into();
        let edges: DataFrame = edges_dataframe.into();

        let node_attribute_columns = nodes.columns(node_attribute_column_names.clone()).unwrap();
        let edge_attribute_columns = edges.columns(edge_attribute_column_names.clone()).unwrap();

        let node_index_column = nodes
            .column(&node_index_column_name)
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter();

        let edge_from_index_column = edges
            .column(&edge_from_index_column_name)
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter();
        let edge_to_index_column = edges
            .column(&edge_to_index_column_name)
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter();

        let edge_index_columns = edge_from_index_column.zip(edge_to_index_column);

        for (index, node_index) in node_index_column.enumerate() {
            let id = node_index.unwrap();
            index_mapping
                .insert_custom_index_to_node_index((*id).to_string(), NodeIndex::new(index));

            let mut weight = Dictionary::new();

            for (column_index, column_name) in node_attribute_column_names.iter().enumerate() {
                let value = Python::with_gil(|py| {
                    PyAnyValue(
                        node_attribute_columns[column_index]
                            .get(index)
                            .unwrap()
                            .clone(),
                    )
                    .into_py(py)
                });
                weight.insert(column_name.clone(), value);
            }

            elements.push(Element::Node { weight });
        }

        for (index, (from_index, to_index)) in edge_index_columns.enumerate() {
            let from_node_index =
                index_mapping
                    .get_node_index(from_index.unwrap())
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        from_index.unwrap()
                    )))?;

            let to_node_index =
                index_mapping
                    .get_node_index(to_index.unwrap())
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        to_index.unwrap()
                    )))?;

            let mut weight = Dictionary::new();

            for (column_index, column_name) in edge_attribute_column_names.iter().enumerate() {
                let value = Python::with_gil(|py| {
                    PyAnyValue(
                        edge_attribute_columns[column_index]
                            .get(index)
                            .unwrap()
                            .clone(),
                    )
                    .into_py(py)
                });
                weight.insert(column_name.clone(), value);
            }

            elements.push(Element::Edge {
                source: from_node_index.index(),
                target: to_node_index.index(),
                weight,
            });
        }

        Ok(Medrecord {
            graph: StableGraph::from_elements(elements),
            index_mapping: index_mapping.to_owned(),
        })
    }

    fn node_count(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }

    fn edge_count(&self) -> PyResult<usize> {
        Ok(self.graph.edge_count())
    }

    #[getter]
    fn nodes(&self) -> PyResult<Vec<&String>> {
        Ok(Vec::from_iter(
            self.index_mapping.custom_index_to_node_index_keys(),
        ))
    }

    #[pyo3(signature = (*node_id))]
    fn node(&self, node_id: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
        node_id
            .iter()
            .map(|item| {
                let id = item.extract::<String>()?;

                let node_index =
                    self.index_mapping
                        .get_node_index(&id)
                        .ok_or(PyIndexError::new_err(format!(
                            "Could not find node with index {}",
                            id
                        )))?;

                let weight = self
                    .graph
                    .node_weight(*node_index)
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find node with index {}",
                        id
                    )))
                    .cloned()?;

                Ok((id, weight))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    #[getter]
    fn edges(&self) -> PyResult<Vec<(String, String)>> {
        self.graph
            .edge_indices()
            .map(|index| {
                let (node_index_start, node_index_end) =
                    self.graph
                        .edge_endpoints(index)
                        .ok_or(PyRuntimeError::new_err(format!(
                            "Unexpected error. Could not find edge with id {}",
                            index.index()
                        )))?;

                let index_start = self
                    .index_mapping
                    .get_custom_index(&node_index_start)
                    .ok_or(PyRuntimeError::new_err(format!(
                        "Unexpected error. Could not find edge with index {}",
                        node_index_start.index()
                    )))?
                    .to_owned();

                let index_end = self
                    .index_mapping
                    .get_custom_index(&node_index_end)
                    .ok_or(PyRuntimeError::new_err(format!(
                        "Unexpected error. Could not find edge with index {}",
                        node_index_end.index()
                    )))?
                    .to_owned();

                Ok((index_start, index_end))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn edges_between(&self, start_node_id: &str, end_node_id: &str) -> PyResult<Vec<Dictionary>> {
        let node_index_start_node =
            self.index_mapping
                .get_node_index(start_node_id)
                .ok_or(PyIndexError::new_err(format!(
                    "Could not find index {}",
                    start_node_id
                )))?;

        let node_index_end_node =
            self.index_mapping
                .get_node_index(end_node_id)
                .ok_or(PyIndexError::new_err(format!(
                    "Could not find index {}",
                    end_node_id
                )))?;

        Ok(self
            .graph
            .edges_connecting(
                node_index_start_node.to_owned(),
                node_index_end_node.to_owned(),
            )
            .map(|edge| edge.weight().to_owned())
            .collect::<Vec<_>>())
    }

    fn add_nodes(&mut self, nodes: Vec<(String, Dictionary)>) -> PyResult<()> {
        for node in nodes.iter() {
            let (id, attributes) = node;

            let node_index = self.graph.add_node(attributes.to_owned());

            self.index_mapping
                .insert_custom_index_to_node_index(id.to_owned(), node_index);
        }

        Ok(())
    }

    fn add_edges(&mut self, relations: Vec<(String, String, Dictionary)>) -> PyResult<()> {
        for relation in relations.iter() {
            let (id_node_1, id_node_2, attributes) = relation;

            let node_index_node_1 =
                self.index_mapping
                    .get_node_index(id_node_1)
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        id_node_1
                    )))?;

            let node_index_node_2 =
                self.index_mapping
                    .get_node_index(id_node_2)
                    .ok_or(PyIndexError::new_err(format!(
                        "Could not find index {}",
                        id_node_2
                    )))?;

            self.graph.add_edge(
                node_index_node_1.to_owned(),
                node_index_node_2.to_owned(),
                attributes.to_owned(),
            );
        }

        Ok(())
    }
}

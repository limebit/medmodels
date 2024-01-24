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
    exceptions::{PyAssertionError, PyIndexError, PyRuntimeError},
    prelude::*,
    types::PyTuple,
    PyTraverseError, PyVisit,
};
use pyo3_polars::PyDataFrame;

type Dictionary = HashMap<String, PyObject>;
type Group = String;
type NodeId = String;

#[pyclass]
pub struct Medrecord {
    graph: StableGraph<Dictionary, Dictionary, Directed>,
    index_mapping: IndexMapping,
    group_mapping: HashMap<Group, Vec<NodeId>>,
}

#[pymethods]
impl Medrecord {
    #[new]
    fn new() -> Self {
        Self {
            graph: StableGraph::default(),
            index_mapping: IndexMapping::new(),
            group_mapping: HashMap::new(),
        }
    }

    #[staticmethod]
    fn from_nodes_and_edges(
        nodes: Vec<(String, Dictionary)>,
        edges: Vec<(String, String, Dictionary)>,
    ) -> PyResult<Self> {
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
            group_mapping: HashMap::new(),
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

        let node_attribute_columns = nodes.columns(&node_attribute_column_names).map_err(|_| {
            PyIndexError::new_err(format!(
                "Could not find all columns from list [{}] in nodes dataframe",
                node_attribute_column_names.join(", ")
            ))
        })?;
        let edge_attribute_columns = edges.columns(&edge_attribute_column_names).map_err(|_| {
            PyIndexError::new_err(format!(
                "Could not find all columns from list [{}] in edges dataframe",
                edge_attribute_column_names.join(", ")
            ))
        })?;

        let node_index_column = nodes
            .column(&node_index_column_name)
            .map_err(|_| {
                PyIndexError::new_err(format!(
                    "Could not find column with name {} in nodes dataframe",
                    node_index_column_name
                ))
            })?
            .utf8()
            .map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "Failed to convert column {} to utf8 in nodes dataframe",
                    node_index_column_name
                ))
            })?
            .into_iter();

        for (index, node_index) in node_index_column.enumerate() {
            let id = node_index.unwrap();
            index_mapping
                .insert_custom_index_to_node_index((*id).to_string(), NodeIndex::new(index));

            let mut weight = Dictionary::new();

            for (column_index, column_name) in node_attribute_column_names.iter().enumerate() {
                let value = Python::with_gil(|py| {
                    PyAnyValue(
                        node_attribute_columns
                            .get(column_index)
                            .unwrap()
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

        let edge_from_index_column = edges
            .column(&edge_from_index_column_name)
            .map_err(|_| {
                PyIndexError::new_err(format!(
                    "Could not find column with name {} in edges dataframe",
                    edge_from_index_column_name
                ))
            })?
            .utf8()
            .map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "Failed to convert column {} to utf8 in edges dataframe",
                    edge_from_index_column_name
                ))
            })?
            .into_iter();
        let edge_to_index_column = edges
            .column(&edge_to_index_column_name)
            .map_err(|_| {
                PyIndexError::new_err(format!(
                    "Could not find column with name {} in edges dataframe",
                    edge_to_index_column_name
                ))
            })?
            .utf8()
            .map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "Failed to convert column {} to utf8 in edges dataframe",
                    edge_to_index_column_name
                ))
            })?
            .into_iter();

        let edge_index_columns = edge_from_index_column.zip(edge_to_index_column);

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
            group_mapping: HashMap::new(),
        })
    }

    fn node_count(&self) -> PyResult<usize> {
        Ok(self.graph.node_count())
    }

    fn edge_count(&self) -> PyResult<usize> {
        Ok(self.graph.edge_count())
    }

    fn group_count(&self) -> PyResult<usize> {
        Ok(self.group_mapping.len())
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

    #[getter]
    fn groups(&self) -> PyResult<Vec<&String>> {
        Ok(self.group_mapping.keys().collect())
    }

    #[pyo3(signature = (*group))]
    fn group(&self, group: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
        group
            .iter()
            .map(|item| {
                let id = item.extract::<String>()?;

                let node_ids =
                    self.group_mapping
                        .get(&id)
                        .ok_or(PyIndexError::new_err(format!(
                            "Could not find group {}",
                            group
                        )))?;

                node_ids
                    .iter()
                    .map(|node_id| {
                        let node_index = self.index_mapping.get_node_index(&node_id).ok_or(
                            PyIndexError::new_err(format!(
                                "Could not find node with index {}",
                                node_id
                            )),
                        )?;

                        let weight = self
                            .graph
                            .node_weight(*node_index)
                            .ok_or(PyIndexError::new_err(format!(
                                "Could not find node with index {}",
                                id
                            )))
                            .cloned()?;

                        Ok((node_id.to_string(), weight))
                    })
                    .collect::<Result<Vec<_>, PyErr>>()
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(|item| Ok(item)).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect::<Result<Vec<_>, _>>()
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

    fn add_group(&mut self, group: Group, node_ids_to_add: Option<Vec<String>>) -> PyResult<()> {
        // Check that the node_ids that are about to be added are actually in the graph
        if let Some(node_ids_to_add) = node_ids_to_add.clone() {
            if node_ids_to_add
                .iter()
                .any(|node_id| !self.index_mapping.check_custom_index(node_id))
            {
                return Err(PyIndexError::new_err(format!(
                    "One or more nodes are not in the graph"
                )));
            }
        }

        self.group_mapping
            .insert(group, node_ids_to_add.unwrap_or_default());

        Ok(())
    }

    fn remove_group(&mut self, group: &str) -> PyResult<()> {
        if !self.group_mapping.contains_key(group) {
            return Err(PyAssertionError::new_err(format!(
                "Could not find group {}",
                group
            )));
        }

        self.group_mapping.remove(group);

        Ok(())
    }

    fn remove_from_group(&mut self, group: Group, node_id: &str) -> PyResult<()> {
        let node_ids = self
            .group_mapping
            .get_mut(&group)
            .ok_or(PyIndexError::new_err(format!(
                "Could not find group {}",
                group
            )))?;

        let node_id_index =
            node_ids
                .iter()
                .position(|id| id == node_id)
                .ok_or(PyIndexError::new_err(format!(
                    "Could not find node with id {} in group {}",
                    node_id, group
                )))?;

        node_ids.remove(node_id_index);

        Ok(())
    }

    #[pyo3(signature = (group, *node_id))]
    fn add_to_group(&mut self, group: Group, node_id: String) -> PyResult<()> {
        let node_ids = self
            .group_mapping
            .get_mut(&group)
            .ok_or(PyIndexError::new_err(format!(
                "Could not find group {}",
                group
            )))?;

        if node_ids.contains(&node_id) {
            return Err(PyAssertionError::new_err(format!(
                "Node with id {} is already in group {}",
                node_id, group
            )));
        }

        node_ids.push(node_id);

        Ok(())
    }

    #[pyo3(signature = (*node_id))]
    fn neighbors(&self, node_id: &PyTuple) -> PyResult<Vec<(String, Dictionary)>> {
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

                let nodes = self
                    .graph
                    .neighbors(*node_index)
                    .map(|node_index| {
                        let custom_index = self
                            .index_mapping
                            .get_custom_index(&node_index)
                            .expect("Node must exist")
                            .to_owned();

                        let weight = self
                            .graph
                            .node_weight(node_index)
                            .expect("Node weigth must exist")
                            .to_owned();

                        (custom_index, weight)
                    })
                    .collect::<Vec<_>>();

                Ok::<_, PyErr>(nodes)
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(|item| Ok(item)).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for weight in self.graph.node_weights() {
            for value in weight.values() {
                visit.call(value)?;
            }
        }

        Ok(())
    }

    fn __clear__(&mut self) {
        self.graph.clear();
    }
}

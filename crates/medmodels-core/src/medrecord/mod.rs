mod index_mapping;
mod values;

pub use self::values::MedRecordValue;
use crate::errors::MedRecordError;
use index_mapping::IndexMapping;
use petgraph::{
    data::{Element, FromElements},
    stable_graph::{NodeIndex, StableGraph},
    Directed,
};
use std::collections::HashMap;

type Dictionary = HashMap<String, MedRecordValue>;
type Group = String;
type NodeId = String;

pub struct MedRecord {
    graph: StableGraph<Dictionary, Dictionary, Directed>,
    index_mapping: IndexMapping,
    group_mapping: HashMap<Group, Vec<NodeId>>,
}

impl MedRecord {
    pub fn new() -> Self {
        Self {
            graph: StableGraph::default(),
            index_mapping: IndexMapping::new(),
            group_mapping: HashMap::new(),
        }
    }

    pub fn from_nodes_and_edges(
        nodes: Vec<(String, Dictionary)>,
        edges: Vec<(String, String, Dictionary)>,
    ) -> Result<Self, MedRecordError> {
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
                    .ok_or(MedRecordError::IndexError(format!(
                        "Could not find index {}",
                        id_node_1
                    )))?;

            let node_index_node_2 =
                index_mapping
                    .get_node_index(id_node_2)
                    .ok_or(MedRecordError::IndexError(format!(
                        "Could not find index {}",
                        id_node_2
                    )))?;

            elements.push(Element::Edge {
                source: node_index_node_1.index(),
                target: node_index_node_2.index(),
                weight: weight.to_owned(),
            });
        }

        Ok(Self {
            graph: StableGraph::from_elements(elements),
            index_mapping: index_mapping.to_owned(),
            group_mapping: HashMap::new(),
        })
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn group_count(&self) -> usize {
        self.group_mapping.len()
    }

    pub fn nodes(&self) -> Vec<&String> {
        Vec::from_iter(self.index_mapping.custom_index_to_node_index_keys())
    }

    pub fn node(&self, node_id: Vec<String>) -> Result<Vec<(String, &Dictionary)>, MedRecordError> {
        node_id
            .iter()
            .map(|id| {
                let node_index =
                    self.index_mapping
                        .get_node_index(id)
                        .ok_or(MedRecordError::IndexError(format!(
                            "Could not find node with index {}",
                            id
                        )))?;

                let weight =
                    self.graph
                        .node_weight(*node_index)
                        .ok_or(MedRecordError::IndexError(format!(
                            "Could not find node with index {}",
                            id
                        )))?;

                Ok((id.to_owned(), weight))
            })
            .collect()
    }

    pub fn edges(&self) -> Result<Vec<(&String, &String)>, MedRecordError> {
        self.graph
            .edge_indices()
            .map(|index| {
                let (node_index_start, node_index_end) =
                    self.graph
                        .edge_endpoints(index)
                        .ok_or(MedRecordError::IndexError(format!(
                            "Unexpected error. Could not find edge with id {}",
                            index.index()
                        )))?;

                let index_start = self
                    .index_mapping
                    .get_custom_index(&node_index_start)
                    .ok_or(MedRecordError::IndexError(format!(
                        "Unexpected error. Could not find edge with index {}",
                        node_index_start.index()
                    )))?;

                let index_end = self.index_mapping.get_custom_index(&node_index_end).ok_or(
                    MedRecordError::IndexError(format!(
                        "Unexpected error. Could not find edge with index {}",
                        node_index_end.index()
                    )),
                )?;

                Ok((index_start, index_end))
            })
            .collect()
    }

    pub fn edges_between(
        &self,
        start_node_id: &str,
        end_node_id: &str,
    ) -> Result<Vec<&Dictionary>, MedRecordError> {
        let node_index_start_node =
            self.index_mapping
                .get_node_index(start_node_id)
                .ok_or(MedRecordError::IndexError(format!(
                    "Could not find index {}",
                    start_node_id
                )))?;

        let node_index_end_node =
            self.index_mapping
                .get_node_index(end_node_id)
                .ok_or(MedRecordError::IndexError(format!(
                    "Could not find index {}",
                    end_node_id
                )))?;

        Ok(self
            .graph
            .edges_connecting(
                node_index_start_node.to_owned(),
                node_index_end_node.to_owned(),
            )
            .map(|edge| edge.weight())
            .collect())
    }

    pub fn groups(&self) -> Vec<&String> {
        self.group_mapping.keys().collect()
    }

    pub fn group(&self, group: Vec<String>) -> Result<Vec<(&String, &Dictionary)>, MedRecordError> {
        group
            .iter()
            .map(|id| {
                let node_ids = self
                    .group_mapping
                    .get(id)
                    .ok_or(MedRecordError::IndexError(format!(
                        "Could not find group {}",
                        id
                    )))?;

                node_ids
                    .iter()
                    .map(|node_id| {
                        let node_index = self.index_mapping.get_node_index(node_id).ok_or(
                            MedRecordError::IndexError(format!(
                                "Could not find node with index {}",
                                node_id
                            )),
                        )?;

                        let weight = self.graph.node_weight(*node_index).ok_or(
                            MedRecordError::IndexError(format!(
                                "Could not find node with index {}",
                                id
                            )),
                        )?;

                        Ok((node_id, weight))
                    })
                    .collect::<Result<Vec<_>, MedRecordError>>()
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(|item| Ok(item)).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect()
    }

    pub fn add_nodes(&mut self, nodes: Vec<(String, Dictionary)>) -> () {
        for node in nodes.iter() {
            let (id, attributes) = node;

            let node_index = self.graph.add_node(attributes.to_owned());

            self.index_mapping
                .insert_custom_index_to_node_index(id.to_owned(), node_index);
        }
    }

    pub fn add_edges(
        &mut self,
        relations: Vec<(String, String, Dictionary)>,
    ) -> Result<(), MedRecordError> {
        for relation in relations.iter() {
            let (id_node_1, id_node_2, attributes) = relation;

            let node_index_node_1 =
                self.index_mapping
                    .get_node_index(id_node_1)
                    .ok_or(MedRecordError::IndexError(format!(
                        "Could not find index {}",
                        id_node_1
                    )))?;

            let node_index_node_2 =
                self.index_mapping
                    .get_node_index(id_node_2)
                    .ok_or(MedRecordError::IndexError(format!(
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

    pub fn add_group(
        &mut self,
        group: Group,
        node_ids_to_add: Option<Vec<String>>,
    ) -> Result<(), MedRecordError> {
        // Check that the node_ids that are about to be added are actually in the graph
        if let Some(node_ids_to_add) = node_ids_to_add.clone() {
            if node_ids_to_add
                .iter()
                .any(|node_id| !self.index_mapping.check_custom_index(node_id))
            {
                return Err(MedRecordError::IndexError(format!(
                    "One or more nodes are not in the graph"
                )));
            }
        }

        self.group_mapping
            .insert(group, node_ids_to_add.unwrap_or_default());

        Ok(())
    }

    pub fn remove_group(&mut self, group: &str) -> Result<(), MedRecordError> {
        if !self.group_mapping.contains_key(group) {
            return Err(MedRecordError::IndexError(format!(
                "Could not find group {}",
                group
            )));
        }

        self.group_mapping.remove(group);

        Ok(())
    }

    pub fn remove_from_group(&mut self, group: Group, node_id: &str) -> Result<(), MedRecordError> {
        let node_ids = self
            .group_mapping
            .get_mut(&group)
            .ok_or(MedRecordError::IndexError(format!(
                "Could not find group {}",
                group
            )))?;

        let node_id_index =
            node_ids
                .iter()
                .position(|id| id == node_id)
                .ok_or(MedRecordError::IndexError(format!(
                    "Could not find node with id {} in group {}",
                    node_id, group
                )))?;

        node_ids.remove(node_id_index);

        Ok(())
    }

    pub fn add_to_group(&mut self, group: Group, node_id: String) -> Result<(), MedRecordError> {
        let node_ids = self
            .group_mapping
            .get_mut(&group)
            .ok_or(MedRecordError::IndexError(format!(
                "Could not find group {}",
                group
            )))?;

        if node_ids.contains(&node_id) {
            return Err(MedRecordError::AssertionError(format!(
                "Node with id {} is already in group {}",
                node_id, group
            )));
        }

        node_ids.push(node_id);

        Ok(())
    }

    pub fn neighbors(
        &self,
        node_id: Vec<String>,
    ) -> Result<Vec<(String, &Dictionary)>, MedRecordError> {
        node_id
            .iter()
            .map(|id| {
                let node_index =
                    self.index_mapping
                        .get_node_index(id)
                        .ok_or(MedRecordError::IndexError(format!(
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
                            .expect("Node weigth must exist");

                        (custom_index, weight)
                    })
                    .collect::<Vec<_>>();

                Ok(nodes)
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(|item| Ok(item)).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect()
    }

    pub fn clear(&mut self) -> () {
        self.graph.clear();
        self.group_mapping.clear();
        self.index_mapping.clear();
    }

    pub fn iter_weights(&self) -> impl Iterator<Item = &Dictionary> {
        self.graph.node_weights()
    }
}

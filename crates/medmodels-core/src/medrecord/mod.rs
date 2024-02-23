mod index_mapping;
mod values;

use crate::{
    errors::MedRecordError,
    polars::{dataframe_to_edges, dataframe_to_nodes},
};
use index_mapping::IndexMapping;
use petgraph::{
    data::{Element, FromElements},
    stable_graph::{NodeIndex, StableGraph},
    Directed,
};
use polars::frame::DataFrame;
use std::collections::HashMap;
pub use values::MedRecordValue;

pub(crate) type Dictionary = HashMap<String, MedRecordValue>;
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

    pub fn from_tuples(
        nodes: Vec<(String, Dictionary)>,
        edges: Option<Vec<(String, String, Dictionary)>>,
    ) -> Result<Self, MedRecordError> {
        let mut index_mapping = IndexMapping::new();

        let edges = edges.unwrap_or_default();

        let mut node_elements = nodes
            .into_iter()
            .enumerate()
            .map(|(i, node)| {
                index_mapping.insert_custom_index_to_node_index(node.0, NodeIndex::new(i));

                Element::<Dictionary, Dictionary>::Node { weight: node.1 }
            })
            .collect::<Vec<_>>();

        let edge_elements = edges
            .into_iter()
            .map(|edge| {
                let from_node_index =
                    index_mapping
                        .get_node_index(&edge.0)
                        .ok_or(MedRecordError::IndexError(format!(
                            "Could not find index {}",
                            edge.0
                        )))?;

                let to_node_index =
                    index_mapping
                        .get_node_index(&edge.1)
                        .ok_or(MedRecordError::IndexError(format!(
                            "Could not find index {}",
                            edge.1
                        )))?;

                Ok(Element::<Dictionary, Dictionary>::Edge {
                    source: from_node_index.index(),
                    target: to_node_index.index(),
                    weight: edge.2,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        node_elements.extend(edge_elements);

        Ok(Self {
            graph: StableGraph::from_elements(node_elements),
            index_mapping,
            group_mapping: HashMap::new(),
        })
    }

    pub fn from_dataframes(
        nodes_dataframe: DataFrame,
        nodes_index_column: &str,
        edges_dataframe: DataFrame,
        edges_from_index_column: &str,
        edges_to_index_column: &str,
    ) -> Result<MedRecord, MedRecordError> {
        let nodes = dataframe_to_nodes(nodes_dataframe, nodes_index_column)?;
        let edges = dataframe_to_edges(
            edges_dataframe,
            edges_from_index_column,
            edges_to_index_column,
        )?;

        Self::from_tuples(nodes, Some(edges))
    }

    pub fn from_nodes_dataframe(
        nodes_dataframe: DataFrame,
        nodes_index_column: &str,
    ) -> Result<MedRecord, MedRecordError> {
        let nodes = dataframe_to_nodes(nodes_dataframe, nodes_index_column)?;

        Self::from_tuples(nodes, None)
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

    pub fn edges(&self) -> Vec<(&String, &String)> {
        self.graph
            .edge_indices()
            .map(|index| {
                let (node_index_start, node_index_end) =
                    self.graph.edge_endpoints(index).expect("Edge must exist");

                let index_from = self
                    .index_mapping
                    .get_custom_index(&node_index_start)
                    .expect("Edge start index must exist");

                let index_to = self
                    .index_mapping
                    .get_custom_index(&node_index_end)
                    .expect("Edge to index must exist");

                (index_from, index_to)
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

    pub fn group(&self, group: Vec<&str>) -> Result<Vec<(&String, &Dictionary)>, MedRecordError> {
        group
            .into_iter()
            .map(|id| {
                let node_ids = self
                    .group_mapping
                    .get(id)
                    .ok_or(MedRecordError::IndexError(format!(
                        "Could not find group {}",
                        id
                    )))?;

                Ok(node_ids
                    .iter()
                    .map(|node_id| {
                        let node_index = self
                            .index_mapping
                            .get_node_index(node_id)
                            .expect("Index must exist");

                        let weight = self
                            .graph
                            .node_weight(*node_index)
                            .expect("Node with index must exist");

                        (node_id, weight)
                    })
                    .collect::<Vec<_>>())
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect()
    }

    pub fn add_node(&mut self, id: String, attributes: Dictionary) {
        let node_index = self.graph.add_node(attributes);

        self.index_mapping
            .insert_custom_index_to_node_index(id, node_index);
    }

    pub fn add_nodes(&mut self, nodes: Vec<(String, Dictionary)>) {
        for (id, attributes) in nodes.into_iter() {
            self.add_node(id, attributes);
        }
    }

    pub fn add_nodes_dataframe(
        &mut self,
        nodes_dataframe: DataFrame,
        index_column_name: &str,
    ) -> Result<(), MedRecordError> {
        let nodes = dataframe_to_nodes(nodes_dataframe, index_column_name)?;

        self.add_nodes(nodes);

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        from_id: String,
        to_id: String,
        attributes: Dictionary,
    ) -> Result<(), MedRecordError> {
        let node_index_node_1 =
            self.index_mapping
                .get_node_index(&from_id)
                .ok_or(MedRecordError::IndexError(format!(
                    "Could not find index {}",
                    from_id
                )))?;

        let node_index_node_2 =
            self.index_mapping
                .get_node_index(&to_id)
                .ok_or(MedRecordError::IndexError(format!(
                    "Could not find index {}",
                    to_id
                )))?;

        self.graph.add_edge(
            node_index_node_1.to_owned(),
            node_index_node_2.to_owned(),
            attributes.to_owned(),
        );

        Ok(())
    }

    pub fn add_edges(
        &mut self,
        edges: Vec<(String, String, Dictionary)>,
    ) -> Result<(), MedRecordError> {
        for (from_id, to_id, attributes) in edges.into_iter() {
            self.add_edge(from_id, to_id, attributes)?
        }

        Ok(())
    }

    pub fn add_edges_dataframe(
        &mut self,
        edges_dataframe: DataFrame,
        from_index_column_name: &str,
        to_index_column_name: &str,
    ) -> Result<(), MedRecordError> {
        let edges = dataframe_to_edges(
            edges_dataframe,
            from_index_column_name,
            to_index_column_name,
        )?;

        self.add_edges(edges)
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
                return Err(MedRecordError::IndexError(
                    "One or more nodes are not in the graph".to_string(),
                ));
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

        if !self.index_mapping.check_custom_index(&node_id) {
            return Err(MedRecordError::IndexError(format!(
                "Could not find node with index {}",
                node_id
            )));
        }

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
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.graph.clear();
        self.group_mapping.clear();
        self.index_mapping.clear();
    }
}

impl Default for MedRecord {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::{MedRecord, MedRecordValue};
    use crate::{errors::MedRecordError, medrecord::Dictionary};
    use polars::prelude::*;
    use std::collections::HashMap;

    fn create_nodes() -> Vec<(String, HashMap<String, MedRecordValue>)> {
        vec![
            (
                "0".to_string(),
                HashMap::from([
                    ("lorem".to_string(), "ipsum".into()),
                    ("dolor".to_string(), "sit".into()),
                ]),
            ),
            (
                "1".to_string(),
                HashMap::from([("amet".to_string(), "consectetur".into())]),
            ),
            (
                "2".to_string(),
                HashMap::from([("adipiscing".to_string(), "elit".into())]),
            ),
            ("3".to_string(), HashMap::new()),
        ]
    }

    fn create_edges() -> Vec<(String, String, HashMap<String, MedRecordValue>)> {
        vec![
            (
                "0".to_string(),
                "1".to_string(),
                HashMap::from([
                    ("sed".to_string(), "do".into()),
                    ("eiusmod".to_string(), "tempor".into()),
                ]),
            ),
            (
                "1".to_string(),
                "2".to_string(),
                HashMap::from([("incididunt".to_string(), "ut".into())]),
            ),
            ("0".to_string(), "2".to_string(), HashMap::new()),
        ]
    }

    fn create_nodes_dataframe() -> Result<DataFrame, PolarsError> {
        let s0 = Series::new("index", &["0", "1"]);
        let s1 = Series::new("attribute", &[1, 2]);
        DataFrame::new(vec![s0, s1])
    }

    fn create_edges_dataframe() -> Result<DataFrame, PolarsError> {
        let s0 = Series::new("from", &["0", "1"]);
        let s1 = Series::new("to", &["1", "0"]);
        let s2 = Series::new("attribute", &[1, 2]);
        DataFrame::new(vec![s0, s1, s2])
    }

    fn create_medrecord() -> MedRecord {
        let nodes = create_nodes();
        let edges = create_edges();

        MedRecord::from_tuples(nodes, Some(edges)).unwrap()
    }

    #[test]
    fn test_from_tuples() {
        let medrecord = create_medrecord();

        assert_eq!(4, medrecord.node_count());
        assert_eq!(3, medrecord.edge_count());
    }

    #[test]
    fn test_invalid_from_tuples() {
        let nodes = create_nodes();

        // Adding an edge pointing to a non-existing node should fail
        assert!(MedRecord::from_tuples(
            nodes.clone(),
            Some(vec![("0".to_string(), "50".to_string(), HashMap::new())])
        )
        .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding an edge from a non-existing should fail
        assert!(MedRecord::from_tuples(
            nodes,
            Some(vec![("50".to_string(), "0".to_string(), HashMap::new())])
        )
        .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_node_count() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        medrecord.add_node("0".to_string(), HashMap::new());

        assert_eq!(1, medrecord.node_count());
    }

    #[test]
    fn test_edge_count() {
        let mut medrecord = MedRecord::new();

        medrecord.add_node("0".to_string(), HashMap::new());
        medrecord.add_node("1".to_string(), HashMap::new());

        assert_eq!(0, medrecord.edge_count());

        medrecord
            .add_edge("0".to_string(), "1".to_string(), HashMap::new())
            .unwrap();

        assert_eq!(1, medrecord.edge_count());
    }

    #[test]
    fn test_group_count() {
        let mut medrecord = create_medrecord();

        assert_eq!(0, medrecord.group_count());

        medrecord.add_group("0".to_string(), None).unwrap();

        assert_eq!(1, medrecord.group_count());
    }

    #[test]
    fn test_nodes() {
        let medrecord = create_medrecord();

        let nodes = create_nodes()
            .into_iter()
            .map(|node| node.0)
            .collect::<Vec<_>>();

        for node in medrecord.nodes() {
            assert!(nodes.contains(node));
        }
    }

    #[test]
    fn test_node() {
        let medrecord = create_medrecord();

        let nodes = medrecord.node(vec!["0".to_string()]).unwrap();

        assert_eq!(1, nodes.len());

        let node = nodes.first().unwrap();

        let mock_nodes = create_nodes();

        let first_node = mock_nodes.first().unwrap();

        assert_eq!("0".to_string(), node.0);
        assert_eq!(first_node.1, *node.1);
    }

    #[test]
    fn test_invalid_node() {
        let medrecord = create_medrecord();

        // Querying a non-existing node should fail
        assert!(medrecord
            .node(vec!["50".to_string()])
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_edges() {
        let medrecord = create_medrecord();
        let edges = create_edges()
            .into_iter()
            .map(|edge| (edge.0, edge.1))
            .collect::<Vec<_>>();

        for edge in medrecord.edges() {
            assert!(edges.contains(&(edge.0.to_owned(), edge.1.to_owned())));
        }
    }

    #[test]
    fn test_edges_between() {
        let medrecord = create_medrecord();

        let edges = medrecord.edges_between("0", "1").unwrap();

        assert_eq!(1, edges.len());
    }

    #[test]
    fn test_invalid_edges_netween() {
        let medrecord = create_medrecord();

        // Querying edges between a existing and a non-existing node should fail
        assert!(medrecord
            .edges_between("0", "50")
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Tests for the other direction
        assert!(medrecord
            .edges_between("50", "0")
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_groups() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".to_string(), None).unwrap();

        let groups = medrecord.groups();

        assert_eq!(vec![&"0".to_string()], groups);
    }

    #[test]
    fn test_group() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".to_string(), None).unwrap();

        let group = medrecord.group(vec!["0"]).unwrap();

        assert_eq!(Vec::<(&String, &Dictionary)>::new(), group);

        medrecord
            .add_group("1".to_string(), Some(vec!["0".to_string()]))
            .unwrap();

        let groups = medrecord.group(vec!["1"]).unwrap();

        let nodes = create_nodes();

        let node = nodes.first().unwrap();

        assert_eq!(1, groups.len());

        let group = groups.first().unwrap();

        assert_eq!(&(&node.0, &node.1), group);
    }

    #[test]
    fn test_invalid_group() {
        let medrecord = create_medrecord();

        // Querying a non-existing group should fail
        assert!(medrecord
            .group(vec!["0"])
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_add_node() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        medrecord.add_node("0".to_string(), HashMap::new());

        assert_eq!(1, medrecord.node_count());
    }

    #[test]
    fn test_add_nodes() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        let nodes = create_nodes();

        medrecord.add_nodes(nodes);

        assert_eq!(4, medrecord.node_count());
    }

    #[test]
    fn test_add_nodes_dataframe() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        let nodes_dataframe = create_nodes_dataframe().unwrap();

        medrecord
            .add_nodes_dataframe(nodes_dataframe, "index")
            .unwrap();

        assert_eq!(2, medrecord.node_count());
    }

    #[test]
    fn test_add_edge() {
        let mut medrecord = create_medrecord();

        assert_eq!(3, medrecord.edge_count());

        medrecord
            .add_edge("0".to_string(), "3".to_string(), HashMap::new())
            .unwrap();

        assert_eq!(4, medrecord.edge_count());
    }

    #[test]
    fn test_invalid_add_edge() {
        let mut medrecord = MedRecord::new();

        let nodes = create_nodes();

        medrecord.add_nodes(nodes);

        // Adding an edge pointing to a non-existing node should fail
        assert!(medrecord
            .add_edge("0".to_string(), "50".to_string(), HashMap::new())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding an edge from a non-existing node should fail
        assert!(medrecord
            .add_edge("50".to_string(), "0".to_string(), HashMap::new())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_add_edges() {
        let mut medrecord = MedRecord::new();

        let nodes = create_nodes();

        medrecord.add_nodes(nodes);

        assert_eq!(0, medrecord.edge_count());

        let edges = create_edges();

        medrecord.add_edges(edges).unwrap();

        assert_eq!(3, medrecord.edge_count());
    }

    #[test]
    fn test_add_edges_dataframe() {
        let mut medrecord = MedRecord::new();

        let nodes = create_nodes();

        medrecord.add_nodes(nodes);

        assert_eq!(0, medrecord.edge_count());

        let edges = create_edges_dataframe().unwrap();

        medrecord.add_edges_dataframe(edges, "from", "to").unwrap();

        assert_eq!(2, medrecord.edge_count());
    }

    #[test]
    fn test_add_group() {
        let mut medrecord = create_medrecord();

        assert_eq!(0, medrecord.group_count());

        medrecord.add_group("0".to_string(), None).unwrap();

        assert_eq!(1, medrecord.group_count());
    }

    #[test]
    fn test_invalid_add_group() {
        let mut medrecord = create_medrecord();

        // Adding a group with a non-existing node should fail
        assert!(medrecord
            .add_group("0".to_string(), Some(vec!["50".to_string()]))
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_remove_group() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".to_string(), None).unwrap();

        assert_eq!(1, medrecord.group_count());

        medrecord.remove_group("0").unwrap();

        assert_eq!(0, medrecord.group_count());
    }

    #[test]
    fn test_invalid_remove_group() {
        let mut medrecord = MedRecord::new();

        // Removing a non-existing group should fail
        assert!(medrecord
            .remove_group("0")
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_remove_from_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group(
                "0".to_string(),
                Some(vec!["0".to_string(), "1".to_string()]),
            )
            .unwrap();

        assert_eq!(2, medrecord.group(vec!["0"]).unwrap().len());

        medrecord.remove_from_group("0".to_string(), "0").unwrap();

        assert_eq!(1, medrecord.group(vec!["0"]).unwrap().len());
    }

    #[test]
    fn test_invalid_remove_from_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".to_string(), Some(vec!["0".to_string()]))
            .unwrap();

        // Removing a node from a non-existing group should fail
        assert!(medrecord
            .remove_from_group("50".to_string(), "0")
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Removing a non-existing node from a group should fail
        assert!(medrecord
            .remove_from_group("0".to_string(), "50")
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_add_to_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group(
                "0".to_string(),
                Some(vec!["0".to_string(), "1".to_string()]),
            )
            .unwrap();

        assert_eq!(2, medrecord.group(vec!["0"]).unwrap().len());

        medrecord
            .add_to_group("0".to_string(), "2".to_string())
            .unwrap();

        assert_eq!(3, medrecord.group(vec!["0"]).unwrap().len());
    }

    #[test]
    fn test_invalid_add_to_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".to_string(), Some(vec!["0".to_string()]))
            .unwrap();

        // Adding to a non-existing group should fail
        assert!(medrecord
            .add_to_group("1".to_string(), "0".to_string())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding a non-existing node to a group should fail
        assert!(medrecord
            .add_to_group("0".to_string(), "50".to_string())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding a node to a group that already is in the group should fail
        assert!(medrecord
            .add_to_group("0".to_string(), "0".to_string())
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_neighbors() {
        let medrecord = create_medrecord();

        let neighbors = medrecord.neighbors(vec!["0".to_string()]).unwrap();

        assert_eq!(2, neighbors.len());

        let neighbors = medrecord
            .neighbors(vec!["0".to_string(), "1".to_string()])
            .unwrap();

        assert_eq!(3, neighbors.len());
    }

    #[test]
    fn test_invalid_neighbors() {
        let medrecord = MedRecord::new();

        // Querying neighbors of a non-existing node sohuld fail
        assert!(medrecord
            .neighbors(vec!["0".to_string()])
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_clear() {
        let mut medrecord = create_medrecord();

        medrecord.clear();

        assert_eq!(0, medrecord.node_count());
        assert_eq!(0, medrecord.edge_count());
        assert_eq!(0, medrecord.group_count());
    }
}

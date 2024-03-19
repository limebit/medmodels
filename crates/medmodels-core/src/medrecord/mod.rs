mod attribute;
mod graph;
mod polars;
mod value;

pub use self::{
    attribute::MedRecordAttribute,
    graph::{Attributes, EdgeIndex, NodeIndex},
    value::MedRecordValue,
};
use crate::errors::MedRecordError;
use ::polars::frame::DataFrame;
use graph::Graph;
use medmodels_utils::aliases::MrHashMap;
use polars::{dataframe_to_edges, dataframe_to_nodes};

pub type Group = MedRecordAttribute;

pub struct MedRecord {
    graph: Graph,
    group_mapping: MrHashMap<Group, Vec<NodeIndex>>,
}

impl MedRecord {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            group_mapping: MrHashMap::new(),
        }
    }

    pub fn from_tuples(
        nodes: Vec<(NodeIndex, Attributes)>,
        edges: Option<Vec<(NodeIndex, NodeIndex, Attributes)>>,
    ) -> Result<Self, MedRecordError> {
        Ok(Self {
            graph: Graph::from_tuples(nodes, edges).map_err(MedRecordError::from)?,
            group_mapping: MrHashMap::new(),
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

    pub fn node_indices(&self) -> impl Iterator<Item = &NodeIndex> {
        self.graph.node_indices()
    }

    pub fn node_attributes(&self, node_index: &NodeIndex) -> Result<&Attributes, MedRecordError> {
        self.graph
            .node_attributes(node_index)
            .map_err(MedRecordError::from)
    }

    pub fn edge_indices(&self) -> impl Iterator<Item = &EdgeIndex> {
        self.graph.edge_indices()
    }

    pub fn edge_attributes(&self, edge_index: &EdgeIndex) -> Result<&Attributes, MedRecordError> {
        self.graph
            .edge_attributes(edge_index)
            .map_err(MedRecordError::from)
    }

    pub fn edges_connecting<'a>(
        &'a self,
        outgoing_node_index: &'a NodeIndex,
        incoming_node_index: &'a NodeIndex,
    ) -> impl Iterator<Item = &EdgeIndex> + 'a {
        self.graph
            .edges_connecting(outgoing_node_index, incoming_node_index)
    }

    pub fn groups(&self) -> impl Iterator<Item = &Group> {
        self.group_mapping.keys()
    }

    pub fn nodes_in_group(
        &self,
        group: &Group,
    ) -> Result<impl Iterator<Item = &NodeIndex>, MedRecordError> {
        Ok(self
            .group_mapping
            .get(group)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find group {}",
                group
            )))?
            .iter())
    }

    pub fn add_node(&mut self, node_index: NodeIndex, attributes: Attributes) {
        self.graph.add_node(node_index, attributes);
    }

    pub fn add_nodes(&mut self, nodes: Vec<(NodeIndex, Attributes)>) {
        for (node_index, attributes) in nodes.into_iter() {
            self.add_node(node_index, attributes);
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
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
        attributes: Attributes,
    ) -> Result<EdgeIndex, MedRecordError> {
        Ok(self
            .graph
            .add_edge(source_node_index, target_node_index, attributes.to_owned())?)
    }

    pub fn add_edges(
        &mut self,
        edges: Vec<(NodeIndex, NodeIndex, Attributes)>,
    ) -> Result<Vec<EdgeIndex>, MedRecordError> {
        edges
            .into_iter()
            .map(|(source_edge_index, target_node_index, attributes)| {
                self.add_edge(source_edge_index, target_node_index, attributes)
            })
            .collect()
    }

    pub fn add_edges_dataframe(
        &mut self,
        edges_dataframe: DataFrame,
        from_index_column_name: &str,
        to_index_column_name: &str,
    ) -> Result<Vec<EdgeIndex>, MedRecordError> {
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
        node_indices_to_add: Option<Vec<NodeIndex>>,
    ) -> Result<(), MedRecordError> {
        let Some(node_indices_to_add) = node_indices_to_add else {
            self.group_mapping.insert(group, vec![]);

            return Ok(());
        };

        // Check that the node_indexs that are about to be added are actually in the graph
        for node_index in &node_indices_to_add {
            if !self.graph.contains_node(node_index) {
                return Err(MedRecordError::IndexError(format!(
                    "Cannot find node with index {}",
                    node_index,
                )));
            }
        }

        self.group_mapping.insert(group, node_indices_to_add);

        Ok(())
    }

    pub fn remove_group(&mut self, group: &Group) -> Result<(), MedRecordError> {
        if !self.group_mapping.contains_key(group) {
            return Err(MedRecordError::IndexError(format!(
                "Cannot find group {}",
                group
            )));
        }

        self.group_mapping.remove(group);

        Ok(())
    }

    pub fn remove_from_group(
        &mut self,
        group: &Group,
        node_index: &NodeIndex,
    ) -> Result<(), MedRecordError> {
        let node_indices = self
            .group_mapping
            .get_mut(group)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find group {}",
                group
            )))?;

        let node_index = node_indices
            .iter()
            .position(|index| index == node_index)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find node with index {} in group {}",
                node_index, group
            )))?;

        node_indices.remove(node_index);

        Ok(())
    }

    pub fn add_to_group(
        &mut self,
        group: &Group,
        node_index: NodeIndex,
    ) -> Result<(), MedRecordError> {
        if !self.node_indices().any(|index| *index == node_index) {
            return Err(MedRecordError::AssertionError(format!(
                "Node with index {} could not be found",
                node_index
            )));
        }

        let node_indices = self
            .group_mapping
            .get_mut(group)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find group {}",
                group
            )))?;

        if node_indices.contains(&node_index) {
            return Err(MedRecordError::AssertionError(format!(
                "Node with index {} is already in group {}",
                node_index, group
            )));
        }

        node_indices.push(node_index);

        Ok(())
    }

    pub fn neighbors(
        &self,
        node_index: &NodeIndex,
    ) -> Result<impl Iterator<Item = &NodeIndex>, MedRecordError> {
        self.graph
            .neighbors(node_index)
            .map_err(MedRecordError::from)
    }

    pub fn clear(&mut self) {
        self.graph.clear();
        self.group_mapping.clear();
    }
}

impl Default for MedRecord {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::{Attributes, MedRecord, MedRecordAttribute};
    use crate::{errors::MedRecordError, medrecord::NodeIndex};
    use polars::prelude::*;
    use std::collections::HashMap;

    fn create_nodes() -> Vec<(NodeIndex, Attributes)> {
        vec![
            (
                "0".into(),
                HashMap::from([
                    ("lorem".into(), "ipsum".into()),
                    ("dolor".into(), "sit".into()),
                ]),
            ),
            (
                "1".into(),
                HashMap::from([("amet".into(), "consectetur".into())]),
            ),
            (
                "2".into(),
                HashMap::from([("adipiscing".into(), "elit".into())]),
            ),
            ("3".into(), HashMap::new()),
        ]
    }

    fn create_edges() -> Vec<(NodeIndex, NodeIndex, Attributes)> {
        vec![
            (
                "0".into(),
                "1".into(),
                HashMap::from([
                    ("sed".into(), "do".into()),
                    ("eiusmod".into(), "tempor".into()),
                ]),
            ),
            (
                "1".into(),
                "2".into(),
                HashMap::from([("incididunt".into(), "ut".into())]),
            ),
            ("0".into(), "2".into(), HashMap::new()),
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
            Some(vec![("0".into(), "50".into(), HashMap::new())])
        )
        .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding an edge from a non-existing should fail
        assert!(MedRecord::from_tuples(
            nodes,
            Some(vec![("50".into(), "0".into(), HashMap::new())])
        )
        .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_node_count() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        medrecord.add_node("0".into(), HashMap::new());

        assert_eq!(1, medrecord.node_count());
    }

    #[test]
    fn test_edge_count() {
        let mut medrecord = MedRecord::new();

        medrecord.add_node("0".into(), HashMap::new());
        medrecord.add_node("1".into(), HashMap::new());

        assert_eq!(0, medrecord.edge_count());

        medrecord
            .add_edge("0".into(), "1".into(), HashMap::new())
            .unwrap();

        assert_eq!(1, medrecord.edge_count());
    }

    #[test]
    fn test_group_count() {
        let mut medrecord = create_medrecord();

        assert_eq!(0, medrecord.group_count());

        medrecord.add_group("0".into(), None).unwrap();

        assert_eq!(1, medrecord.group_count());
    }

    #[test]
    fn test_node_indices() {
        let medrecord = create_medrecord();

        let node_indices = create_nodes()
            .into_iter()
            .map(|(node_index, _)| node_index)
            .collect::<Vec<_>>();

        for node_index in medrecord.node_indices() {
            assert!(node_indices.contains(node_index));
        }
    }

    #[test]
    fn test_node_attributes() {
        let medrecord = create_medrecord();

        let attributes = medrecord.node_attributes(&"0".into()).unwrap();

        assert_eq!(&create_nodes()[0].1, attributes);
    }

    #[test]
    fn test_invalid_node_attributes() {
        let medrecord = create_medrecord();

        // Querying a non-existing node should fail
        assert!(medrecord
            .node_attributes(&"50".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_edge_indices() {
        let medrecord = create_medrecord();
        let edges = [0, 1, 2];

        for edge in medrecord.edge_indices() {
            assert!(edges.contains(edge));
        }
    }

    #[test]
    fn test_edge_attributes() {
        let medrecord = create_medrecord();

        let attributes = medrecord.edge_attributes(&0).unwrap();

        assert_eq!(&create_edges()[0].2, attributes);
    }

    #[test]
    fn test_invalid_edge_attributes() {
        let medrecord = create_medrecord();

        // Querying a non-existing node should fail
        assert!(medrecord
            .edge_attributes(&50)
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_edges_connecting() {
        let medrecord = create_medrecord();

        let first_index = "0".into();
        let second_index = "1".into();
        let edges_connecting = medrecord.edges_connecting(&first_index, &second_index);

        assert_eq!(vec![&0], edges_connecting.collect::<Vec<_>>());

        let first_index = "0".into();
        let second_index = "3".into();
        let edges_connecting = medrecord.edges_connecting(&first_index, &second_index);

        assert_eq!(0, edges_connecting.count());
    }

    #[test]
    fn test_groups() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".into(), None).unwrap();

        let groups = medrecord.groups().collect::<Vec<_>>();

        assert_eq!(vec![&(MedRecordAttribute::from("0"))], groups);
    }

    #[test]
    fn test_nodes_in_group() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".into(), None).unwrap();

        let group = medrecord.nodes_in_group(&"0".into()).unwrap();

        assert_eq!(0, group.count());

        medrecord
            .add_group("1".into(), Some(vec!["0".into()]))
            .unwrap();

        let groups = medrecord.nodes_in_group(&"1".into()).unwrap();

        let nodes = create_nodes();

        let node = nodes.first().unwrap();

        let groups = groups.collect::<Vec<_>>();
        assert_eq!(1, groups.len());

        let group = groups.first().unwrap();

        assert_eq!(&node.0, *group);
    }

    #[test]
    fn test_invalid_nodes_in_group() {
        let medrecord = create_medrecord();

        // Querying a non-existing group should fail
        assert!(medrecord
            .nodes_in_group(&"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_add_node() {
        let mut medrecord = MedRecord::new();

        assert_eq!(0, medrecord.node_count());

        medrecord.add_node("0".into(), HashMap::new());

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
            .add_edge("0".into(), "3".into(), HashMap::new())
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
            .add_edge("0".into(), "50".into(), HashMap::new())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding an edge from a non-existing node should fail
        assert!(medrecord
            .add_edge("50".into(), "0".into(), HashMap::new())
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

        medrecord.add_group("0".into(), None).unwrap();

        assert_eq!(1, medrecord.group_count());
    }

    #[test]
    fn test_invalid_add_group() {
        let mut medrecord = create_medrecord();

        // Adding a group with a non-existing node should fail
        assert!(medrecord
            .add_group("0".into(), Some(vec!["50".into()]))
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_remove_group() {
        let mut medrecord = create_medrecord();

        medrecord.add_group("0".into(), None).unwrap();

        assert_eq!(1, medrecord.group_count());

        medrecord.remove_group(&"0".into()).unwrap();

        assert_eq!(0, medrecord.group_count());
    }

    #[test]
    fn test_invalid_remove_group() {
        let mut medrecord = MedRecord::new();

        // Removing a non-existing group should fail
        assert!(medrecord
            .remove_group(&"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_remove_from_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]))
            .unwrap();

        assert_eq!(2, medrecord.nodes_in_group(&"0".into()).unwrap().count());

        medrecord
            .remove_from_group(&"0".into(), &"0".into())
            .unwrap();

        assert_eq!(1, medrecord.nodes_in_group(&"0".into()).unwrap().count());
    }

    #[test]
    fn test_invalid_remove_from_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".into(), Some(vec!["0".into()]))
            .unwrap();

        // Removing a node from a non-existing group should fail
        assert!(medrecord
            .remove_from_group(&"50".into(), &"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Removing a non-existing node from a group should fail
        assert!(medrecord
            .remove_from_group(&"0".into(), &"50".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_add_to_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]))
            .unwrap();

        assert_eq!(2, medrecord.nodes_in_group(&"0".into()).unwrap().count());

        medrecord.add_to_group(&"0".into(), "2".into()).unwrap();

        assert_eq!(3, medrecord.nodes_in_group(&"0".into()).unwrap().count());
    }

    #[test]
    fn test_invalid_add_to_group() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("0".into(), Some(vec!["0".into()]))
            .unwrap();

        // Adding to a non-existing group should fail
        assert!(medrecord
            .add_to_group(&"1".into(), "0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding a non-existing node to a group should fail
        assert!(medrecord
            .add_to_group(&"0".into(), "50".into())
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        // Adding a node to a group that already is in the group should fail
        assert!(medrecord
            .add_to_group(&"0".into(), "0".into())
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_neighbors() {
        let medrecord = create_medrecord();

        let neighbors = medrecord.neighbors(&"0".into()).unwrap();

        assert_eq!(2, neighbors.count());
    }

    #[test]
    fn test_invalid_neighbors() {
        let medrecord = MedRecord::new();

        // Querying neighbors of a non-existing node sohuld fail
        assert!(medrecord
            .neighbors(&"0".into())
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

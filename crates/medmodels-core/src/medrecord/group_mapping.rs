use super::{EdgeIndex, MedRecordAttribute, NodeIndex};
use crate::errors::MedRecordError;
use medmodels_utils::aliases::{MrHashMap, MrHashMapEntry, MrHashSet};
use serde::{Deserialize, Serialize};

pub type Group = MedRecordAttribute;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(super) struct GroupMapping {
    pub(super) nodes_in_group: MrHashMap<Group, MrHashSet<NodeIndex>>,
    pub(super) edges_in_group: MrHashMap<Group, MrHashSet<EdgeIndex>>,
    pub(super) groups_of_node: MrHashMap<NodeIndex, MrHashSet<Group>>,
    pub(super) groups_of_edge: MrHashMap<EdgeIndex, MrHashSet<Group>>,
}

impl GroupMapping {
    pub fn new() -> Self {
        Self {
            nodes_in_group: MrHashMap::new(),
            edges_in_group: MrHashMap::new(),
            groups_of_node: MrHashMap::new(),
            groups_of_edge: MrHashMap::new(),
        }
    }

    pub fn add_group(
        &mut self,
        group: Group,
        node_indices: Option<Vec<NodeIndex>>,
        edge_indices: Option<Vec<EdgeIndex>>,
    ) -> Result<(), MedRecordError> {
        match self.nodes_in_group.entry(group.clone()) {
            MrHashMapEntry::Occupied(o) => Err(MedRecordError::AssertionError(format!(
                "Group {} already exists",
                o.key()
            ))),
            MrHashMapEntry::Vacant(v) => {
                v.insert(MrHashSet::from_iter(
                    node_indices.clone().unwrap_or_default().into_iter(),
                ));
                Ok(())
            }
        }?;

        match self.edges_in_group.entry(group.clone()) {
            MrHashMapEntry::Occupied(o) => Err(MedRecordError::AssertionError(format!(
                "Group {} already exists",
                o.key()
            ))),
            MrHashMapEntry::Vacant(v) => {
                v.insert(MrHashSet::from_iter(
                    edge_indices.clone().unwrap_or_default().into_iter(),
                ));
                Ok(())
            }
        }?;

        match (node_indices, edge_indices) {
            (None, None) => (),
            (None, Some(edge_indices)) => {
                for edge_index in edge_indices {
                    self.groups_of_edge
                        .entry(edge_index)
                        .or_default()
                        .insert(group.clone());
                }
            }
            (Some(node_indices), None) => {
                for node_index in node_indices {
                    self.groups_of_node
                        .entry(node_index)
                        .or_default()
                        .insert(group.clone());
                }
            }
            (Some(node_indices), Some(edge_indices)) => {
                for node_index in node_indices {
                    self.groups_of_node
                        .entry(node_index)
                        .or_default()
                        .insert(group.clone());
                }

                for edge_index in edge_indices {
                    self.groups_of_edge
                        .entry(edge_index)
                        .or_default()
                        .insert(group.clone());
                }
            }
        };

        Ok(())
    }

    pub fn add_node_to_group(
        &mut self,
        group: Group,
        node_index: NodeIndex,
    ) -> Result<(), MedRecordError> {
        let nodes_in_group =
            self.nodes_in_group
                .get_mut(&group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        if !nodes_in_group.insert(node_index.clone()) {
            return Err(MedRecordError::AssertionError(format!(
                "Node with index {node_index} already in group {group}"
            )));
        }

        self.groups_of_node
            .entry(node_index)
            .or_default()
            .insert(group);

        Ok(())
    }

    pub fn add_edge_to_group(
        &mut self,
        group: Group,
        edge_index: EdgeIndex,
    ) -> Result<(), MedRecordError> {
        let edges_in_group =
            self.edges_in_group
                .get_mut(&group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        if !edges_in_group.insert(edge_index) {
            return Err(MedRecordError::AssertionError(format!(
                "Edge with index {edge_index} already in group {group}"
            )));
        }

        self.groups_of_edge
            .entry(edge_index)
            .or_default()
            .insert(group);

        Ok(())
    }

    pub fn remove_group(&mut self, group: &Group) -> Result<(), MedRecordError> {
        let nodes_in_group =
            self.nodes_in_group
                .remove(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        for node in nodes_in_group {
            self.groups_of_node
                .get_mut(&node)
                .expect("Node must exist")
                .remove(group);
        }

        let edges_in_group =
            self.edges_in_group
                .remove(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        for edge in edges_in_group {
            self.groups_of_edge
                .get_mut(&edge)
                .expect("Edge must exist")
                .remove(group);
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node_index: &NodeIndex) {
        let groups_of_node = self.groups_of_node.remove(node_index);

        let Some(groups_of_node) = groups_of_node else {
            return;
        };

        for group in groups_of_node {
            self.nodes_in_group
                .get_mut(&group)
                .expect("Group must exist")
                .remove(node_index);
        }
    }

    pub fn remove_edge(&mut self, edge_index: &EdgeIndex) {
        let groups_of_edge = self.groups_of_edge.remove(edge_index);

        let Some(groups_of_edge) = groups_of_edge else {
            return;
        };

        for group in groups_of_edge {
            self.edges_in_group
                .get_mut(&group)
                .expect("Group must exist")
                .remove(edge_index);
        }
    }

    pub fn remove_node_from_group(
        &mut self,
        group: &Group,
        node_index: &NodeIndex,
    ) -> Result<(), MedRecordError> {
        let nodes_in_group =
            self.nodes_in_group
                .get_mut(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        nodes_in_group
            .remove(node_index)
            .then_some(())
            .ok_or(MedRecordError::AssertionError(format!(
                "Node with index {node_index} not in group {group}"
            )))
    }

    pub fn remove_edge_from_group(
        &mut self,
        group: &Group,
        edge_index: &EdgeIndex,
    ) -> Result<(), MedRecordError> {
        let edges_in_group =
            self.edges_in_group
                .get_mut(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {group}"
                )))?;

        edges_in_group
            .remove(edge_index)
            .then_some(())
            .ok_or(MedRecordError::AssertionError(format!(
                "Edge with index {edge_index} not in group {group}"
            )))
    }

    pub fn groups(&self) -> impl Iterator<Item = &Group> {
        self.nodes_in_group.keys()
    }

    pub fn nodes_in_group(
        &self,
        group: &Group,
    ) -> Result<impl Iterator<Item = &NodeIndex>, MedRecordError> {
        Ok(self
            .nodes_in_group
            .get(group)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find group {group}"
            )))?
            .iter())
    }

    pub fn edges_in_group(
        &self,
        group: &Group,
    ) -> Result<impl Iterator<Item = &EdgeIndex>, MedRecordError> {
        Ok(self
            .edges_in_group
            .get(group)
            .ok_or(MedRecordError::IndexError(format!(
                "Cannot find group {group}"
            )))?
            .iter())
    }

    pub fn groups_of_node(&self, node_index: &NodeIndex) -> impl Iterator<Item = &Group> {
        self.groups_of_node.get(node_index).into_iter().flatten()
    }

    pub fn groups_of_edge(&self, edge_index: &EdgeIndex) -> impl Iterator<Item = &Group> {
        self.groups_of_edge.get(edge_index).into_iter().flatten()
    }

    pub fn group_count(&self) -> usize {
        self.nodes_in_group.len()
    }

    pub fn contains_group(&self, group: &Group) -> bool {
        self.nodes_in_group.contains_key(group)
    }

    pub fn clear(&mut self) {
        self.nodes_in_group.clear();
        self.edges_in_group.clear();
        self.groups_of_node.clear();
        self.groups_of_edge.clear();
    }
}

#[cfg(test)]
mod test {
    use super::GroupMapping;
    use crate::errors::MedRecordError;

    #[test]
    fn test_add_group() {
        let mut group_mapping = GroupMapping::new();

        assert_eq!(0, group_mapping.group_count());

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(1, group_mapping.group_count());

        group_mapping
            .add_group(
                "1".into(),
                Some(vec!["0".into(), "1".into()]),
                Some(vec![0, 1]),
            )
            .unwrap();

        assert_eq!(2, group_mapping.group_count());
        assert_eq!(
            2,
            group_mapping.nodes_in_group(&"1".into()).unwrap().count()
        );
        assert_eq!(
            2,
            group_mapping.edges_in_group(&"1".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_add_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        // Adding an already existing group should fail
        assert!(group_mapping
            .add_group("0".into(), None, None)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_add_node_to_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(
            0,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );

        group_mapping
            .add_node_to_group("0".into(), "0".into())
            .unwrap();

        assert_eq!(
            1,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_add_node_to_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]), None)
            .unwrap();

        // Adding to a non-existing group should fail
        assert!(group_mapping
            .add_node_to_group("50".into(), "1".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding a node to a group that already is in the group should fail
        assert!(group_mapping
            .add_node_to_group("0".into(), "0".into())
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_add_edge_to_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(
            0,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );

        group_mapping.add_edge_to_group("0".into(), 0).unwrap();

        assert_eq!(
            1,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_add_edge_to_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0]))
            .unwrap();

        // Adding to a non-existing group should fail
        assert!(group_mapping
            .add_edge_to_group("50".into(), 1)
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Adding an edge to a group that already is in the group should fail
        assert!(group_mapping
            .add_edge_to_group("0".into(), 0)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_remove_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(1, group_mapping.group_count());

        group_mapping.remove_group(&"0".into()).unwrap();

        assert_eq!(0, group_mapping.group_count());
    }

    #[test]
    fn test_invalid_remove_group() {
        let mut group_mapping = GroupMapping::new();

        // Removing a non-existing group should fail
        assert!(group_mapping
            .remove_group(&"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_remove_node() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]), None)
            .unwrap();

        assert_eq!(
            1,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );

        group_mapping.remove_node(&"0".into());

        assert_eq!(
            0,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_remove_edge() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0]))
            .unwrap();

        assert_eq!(
            1,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );

        group_mapping.remove_edge(&0);

        assert_eq!(
            0,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_remove_node_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]), None)
            .unwrap();

        assert_eq!(
            2,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );

        group_mapping
            .remove_node_from_group(&"0".into(), &"0".into())
            .unwrap();

        assert_eq!(
            1,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_remove_node_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]), None)
            .unwrap();

        // Removing a node from a non-existing group should fail
        assert!(group_mapping
            .remove_node_from_group(&"50".into(), &"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Removing a non-existing node from a group should fail
        assert!(group_mapping
            .remove_node_from_group(&"0".into(), &"50".into())
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_remove_edge_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0, 1]))
            .unwrap();

        assert_eq!(
            2,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );

        group_mapping
            .remove_edge_from_group(&"0".into(), &0)
            .unwrap();

        assert_eq!(
            1,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_remove_edge_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0]))
            .unwrap();

        // Removing an edge from a non-existing group should fail
        assert!(group_mapping
            .remove_edge_from_group(&"50".into(), &0)
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));

        // Removing a non-existing edge from a group should fail
        assert!(group_mapping
            .remove_edge_from_group(&"0".into(), &50)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_groups() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(1, group_mapping.groups().count());
    }

    #[test]
    fn test_nodes_in_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]), None)
            .unwrap();

        assert_eq!(
            2,
            group_mapping.nodes_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_nodes_in_group() {
        let group_mapping = GroupMapping::new();

        // Querying the nodes in a non-existing group should fail
        assert!(group_mapping
            .nodes_in_group(&"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_edges_in_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0, 1]))
            .unwrap();

        assert_eq!(
            2,
            group_mapping.edges_in_group(&"0".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_edges_in_group() {
        let group_mapping = GroupMapping::new();

        // Querying the edges in a non-existing group should fail
        assert!(group_mapping
            .edges_in_group(&"0".into())
            .is_err_and(|e| matches!(e, MedRecordError::IndexError(_))));
    }

    #[test]
    fn test_groups_of_node() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]), None)
            .unwrap();

        assert_eq!(1, group_mapping.groups_of_node(&"0".into()).count());
    }

    #[test]
    fn test_groups_of_edge() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), None, Some(vec![0]))
            .unwrap();

        assert_eq!(1, group_mapping.groups_of_edge(&0).count());
    }

    #[test]
    fn test_group_count() {
        let mut group_mapping = GroupMapping::new();

        assert_eq!(0, group_mapping.group_count());

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(1, group_mapping.group_count());
    }

    #[test]
    fn test_contains_group() {
        let mut group_mapping = GroupMapping::new();

        assert!(!group_mapping.contains_group(&"0".into()));

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert!(group_mapping.contains_group(&"0".into()));
    }

    #[test]
    fn test_clear() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None, None).unwrap();

        assert_eq!(1, group_mapping.group_count());

        group_mapping.clear();

        assert_eq!(0, group_mapping.group_count());
    }
}

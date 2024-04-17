use super::{MedRecordAttribute, NodeIndex};
use crate::errors::MedRecordError;
use medmodels_utils::aliases::{MrHashMap, MrHashMapEntry, MrHashSet};
use serde::{Deserialize, Serialize};

pub type Group = MedRecordAttribute;

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct GroupMapping {
    nodes_in_group: MrHashMap<Group, MrHashSet<NodeIndex>>,
    groups_of_node: MrHashMap<NodeIndex, MrHashSet<Group>>,
}

impl GroupMapping {
    pub fn new() -> Self {
        Self {
            nodes_in_group: MrHashMap::new(),
            groups_of_node: MrHashMap::new(),
        }
    }

    pub fn add_group(
        &mut self,
        group: Group,
        node_indices: Option<Vec<NodeIndex>>,
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

        let Some(node_indices) = node_indices else {
            return Ok(());
        };

        for node_index in node_indices {
            self.groups_of_node
                .entry(node_index)
                .or_default()
                .insert(group.clone());
        }

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
                    "Cannot find group {}",
                    group
                )))?;

        if !nodes_in_group.insert(node_index.clone()) {
            return Err(MedRecordError::AssertionError(format!(
                "Node with index {} already in group {}",
                node_index, group
            )));
        }

        self.groups_of_node
            .entry(node_index)
            .or_default()
            .insert(group);

        Ok(())
    }

    pub fn remove_group(&mut self, group: &Group) -> Result<(), MedRecordError> {
        let nodes_in_group =
            self.nodes_in_group
                .remove(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {}",
                    group
                )))?;

        for node in nodes_in_group {
            self.groups_of_node
                .get_mut(&node)
                .expect("Node must exist")
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

    pub fn remove_node_from_group(
        &mut self,
        group: &Group,
        node_index: &NodeIndex,
    ) -> Result<(), MedRecordError> {
        let nodes_in_group =
            self.nodes_in_group
                .get_mut(group)
                .ok_or(MedRecordError::IndexError(format!(
                    "Cannot find group {}",
                    group
                )))?;

        nodes_in_group
            .remove(node_index)
            .then_some(())
            .ok_or(MedRecordError::AssertionError(format!(
                "Node with index {} not in group {}",
                node_index, group
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
                "Cannot find group {}",
                group
            )))?
            .iter())
    }

    pub fn groups_of_node(&self, node_index: &NodeIndex) -> impl Iterator<Item = &Group> {
        self.groups_of_node.get(node_index).into_iter().flatten()
    }

    pub fn group_count(&self) -> usize {
        self.nodes_in_group.len()
    }

    pub fn contains_group(&self, group: &Group) -> bool {
        self.nodes_in_group.contains_key(group)
    }

    pub fn clear(&mut self) {
        self.nodes_in_group.clear();
        self.groups_of_node.clear();
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

        group_mapping.add_group("0".into(), None).unwrap();

        assert_eq!(1, group_mapping.group_count());

        group_mapping
            .add_group("1".into(), Some(vec!["0".into(), "1".into()]))
            .unwrap();

        assert_eq!(2, group_mapping.group_count());

        assert_eq!(
            2,
            group_mapping.nodes_in_group(&"1".into()).unwrap().count()
        );
    }

    #[test]
    fn test_invalid_add_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None).unwrap();

        // Adding an already existing group should fail
        assert!(group_mapping
            .add_group("0".into(), None)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_add_node_to_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None).unwrap();

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
            .add_group("0".into(), Some(vec!["0".into()]))
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
    fn test_remove_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None).unwrap();

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
            .add_group("0".into(), Some(vec!["0".into()]))
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
    fn test_remove_node_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]))
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
    fn test_invalid_remove_from_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]))
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
    fn test_groups() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None).unwrap();

        assert_eq!(1, group_mapping.groups().count());
    }

    #[test]
    fn test_nodes_in_group() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into(), "1".into()]))
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
    fn test_groups_of_node() {
        let mut group_mapping = GroupMapping::new();

        group_mapping
            .add_group("0".into(), Some(vec!["0".into()]))
            .unwrap();

        assert_eq!(1, group_mapping.groups_of_node(&"0".into()).count());
    }

    #[test]
    fn test_group_count() {
        let mut group_mapping = GroupMapping::new();

        assert_eq!(0, group_mapping.group_count());

        group_mapping.add_group("0".into(), None).unwrap();

        assert_eq!(1, group_mapping.group_count());
    }

    #[test]
    fn test_contains_group() {
        let mut group_mapping = GroupMapping::new();

        assert!(!group_mapping.contains_group(&"0".into()));

        group_mapping.add_group("0".into(), None).unwrap();

        assert!(group_mapping.contains_group(&"0".into()));
    }

    #[test]
    fn test_clear() {
        let mut group_mapping = GroupMapping::new();

        group_mapping.add_group("0".into(), None).unwrap();

        assert_eq!(1, group_mapping.group_count());

        group_mapping.clear();

        assert_eq!(0, group_mapping.group_count());
    }
}

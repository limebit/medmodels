use std::collections::{hash_map::Keys, HashMap};

use petgraph::stable_graph::NodeIndex;

#[derive(Clone)]
pub struct IndexMapping {
    custom_index_to_node_index_mapping: HashMap<String, NodeIndex>,
    node_index_to_custom_index_mapping: HashMap<NodeIndex, String>,
}

#[allow(dead_code)]
impl IndexMapping {
    pub fn new() -> IndexMapping {
        IndexMapping {
            custom_index_to_node_index_mapping: HashMap::new(),
            node_index_to_custom_index_mapping: HashMap::new(),
        }
    }

    pub fn check_custom_index(&self, custom_index: &str) -> bool {
        self.custom_index_to_node_index_mapping
            .contains_key(custom_index)
    }

    pub fn check_node_index(&self, node_index: &NodeIndex) -> bool {
        self.node_index_to_custom_index_mapping
            .contains_key(node_index)
    }

    pub fn get_node_index(&self, custom_index: &str) -> Option<&NodeIndex> {
        self.custom_index_to_node_index_mapping.get(custom_index)
    }

    pub fn get_custom_index(&self, node_index: &NodeIndex) -> Option<&String> {
        self.node_index_to_custom_index_mapping.get(node_index)
    }

    pub fn insert_custom_index_to_node_index(
        &mut self,
        custom_index: String,
        node_index: NodeIndex,
    ) -> Option<(String, NodeIndex)> {
        let return_nodex_index = self
            .custom_index_to_node_index_mapping
            .insert(custom_index.to_owned(), node_index);
        let return_custom_index = self
            .node_index_to_custom_index_mapping
            .insert(node_index, custom_index.to_owned());

        match (return_custom_index, return_nodex_index) {
            (Some(return_custom_index), Some(return_nodex_index)) => {
                Some((return_custom_index, return_nodex_index))
            }
            _ => None,
        }
    }

    pub fn insert_node_index_to_custom_index(
        &mut self,
        node_index: NodeIndex,
        custom_index: String,
    ) -> Option<(NodeIndex, String)> {
        let return_custom_index = self
            .node_index_to_custom_index_mapping
            .insert(node_index, custom_index.to_owned());
        let return_node_index = self
            .custom_index_to_node_index_mapping
            .insert(custom_index.to_owned(), node_index);

        match (return_node_index, return_custom_index) {
            (Some(return_node_index), Some(return_custom_index)) => {
                Some((return_node_index, return_custom_index))
            }
            _ => None,
        }
    }

    pub fn custom_index_to_node_index_keys(&self) -> Keys<'_, String, NodeIndex> {
        self.custom_index_to_node_index_mapping.keys()
    }

    pub fn node_index_to_custom_index_keys(&self) -> Keys<'_, NodeIndex, String> {
        self.node_index_to_custom_index_mapping.keys()
    }
}

use super::{Attributes, NodeIndex};

#[derive(Debug, Clone)]
pub struct Edge {
    pub attributes: Attributes,
    pub(super) source_node_index: NodeIndex,
    pub(super) target_node_index: NodeIndex,
}

impl Edge {
    pub fn new(
        attributes: Attributes,
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
    ) -> Self {
        Self {
            attributes,
            source_node_index,
            target_node_index,
        }
    }
}

use super::{Attributes, NodeIndex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub(crate) attributes: Attributes,
    pub(crate) source_node_index: NodeIndex,
    pub(crate) target_node_index: NodeIndex,
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

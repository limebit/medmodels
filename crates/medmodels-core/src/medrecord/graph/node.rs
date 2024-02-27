use std::collections::HashSet;

use super::{Attributes, EdgeIndex};

#[derive(Debug, Clone)]
pub struct Node {
    pub attributes: Attributes,
    pub(super) outgoing_edge_indices: HashSet<EdgeIndex>,
    pub(super) incoming_edge_indices: HashSet<EdgeIndex>,
}

impl Node {
    pub fn new(attributes: Attributes) -> Self {
        Self {
            attributes,
            outgoing_edge_indices: HashSet::new(),
            incoming_edge_indices: HashSet::new(),
        }
    }
}

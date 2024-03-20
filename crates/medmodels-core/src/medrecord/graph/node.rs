use super::{Attributes, EdgeIndex};
use medmodels_utils::aliases::MrHashSet;

#[derive(Debug, Clone)]
pub struct Node {
    pub attributes: Attributes,
    pub(super) outgoing_edge_indices: MrHashSet<EdgeIndex>,
    pub(super) incoming_edge_indices: MrHashSet<EdgeIndex>,
}

impl Node {
    pub fn new(attributes: Attributes) -> Self {
        Self {
            attributes,
            outgoing_edge_indices: MrHashSet::new(),
            incoming_edge_indices: MrHashSet::new(),
        }
    }
}

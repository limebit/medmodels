use super::operation::{EdgeOperation, NodeOperation, Operation};
use crate::medrecord::{EdgeIndex, MedRecord, NodeIndex};

pub struct NodeSelection<'a> {
    medrecord: &'a MedRecord,
}

impl<'a> NodeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord) -> Self {
        Self { medrecord }
    }

    pub fn r#where(self, operation: NodeOperation) -> impl Iterator<Item = &'a NodeIndex> {
        operation.evaluate(self.medrecord, self.medrecord.node_indices())
    }
}

pub struct EdgeSelection<'a> {
    medrecord: &'a MedRecord,
}

impl<'a> EdgeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord) -> Self {
        Self { medrecord }
    }

    pub fn r#where(self, operation: EdgeOperation) -> impl Iterator<Item = &'a EdgeIndex> {
        operation.evaluate(self.medrecord, self.medrecord.edge_indices())
    }
}

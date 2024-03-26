use super::operation::{EdgeOperation, NodeOperation, Operation};
use crate::medrecord::{EdgeIndex, MedRecord, NodeIndex};

#[derive(Debug)]
pub struct NodeSelection<'a> {
    medrecord: &'a MedRecord,
    operation: NodeOperation,
}

impl<'a> NodeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord, operation: NodeOperation) -> Self {
        Self {
            medrecord,
            operation,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a NodeIndex> {
        self.operation
            .evaluate(self.medrecord, self.medrecord.node_indices())
    }

    pub fn collect<B: FromIterator<&'a NodeIndex>>(self) -> B {
        FromIterator::from_iter(self.iter())
    }
}

#[derive(Debug)]
pub struct EdgeSelection<'a> {
    medrecord: &'a MedRecord,
    operation: EdgeOperation,
}

impl<'a> EdgeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord, operation: EdgeOperation) -> Self {
        Self {
            medrecord,
            operation,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a EdgeIndex> {
        self.operation
            .evaluate(self.medrecord, self.medrecord.edge_indices())
    }

    pub fn collect<B: FromIterator<&'a EdgeIndex>>(self) -> B {
        FromIterator::from_iter(self.iter())
    }
}

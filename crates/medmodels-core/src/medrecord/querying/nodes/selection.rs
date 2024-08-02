use super::NodeOperandWrapper;
use crate::medrecord::{MedRecord, NodeIndex};

pub struct NodeSelection<'a> {
    medrecord: &'a MedRecord,
    operand: NodeOperandWrapper,
}

impl<'a> NodeSelection<'a> {
    pub fn new<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut NodeOperandWrapper),
    {
        let mut operand = NodeOperandWrapper::new();

        query(&mut operand);

        Self { medrecord, operand }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a NodeIndex> {
        self.operand.evaluate(self.medrecord)
    }

    pub fn collect<B>(self) -> B
    where
        B: FromIterator<&'a NodeIndex>,
    {
        FromIterator::from_iter(self.iter())
    }
}

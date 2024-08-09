use super::NodeOperand;
use crate::medrecord::{
    querying::{evaluate::EvaluateOperand, wrapper::Wrapper},
    MedRecord, NodeIndex,
};

#[derive(Debug, Clone)]
pub struct NodeSelection<'a> {
    medrecord: &'a MedRecord,
    operand: Wrapper<NodeOperand>,
}

impl<'a> NodeSelection<'a> {
    pub fn new<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut operand = Wrapper::<NodeOperand>::new();

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

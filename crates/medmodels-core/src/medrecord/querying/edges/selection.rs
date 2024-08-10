use super::EdgeOperand;
use crate::medrecord::{
    querying::{traits::EvaluateOperand, wrapper::Wrapper},
    EdgeIndex, MedRecord,
};

#[derive(Debug, Clone)]
pub struct EdgeSelection<'a> {
    medrecord: &'a MedRecord,
    operand: Wrapper<EdgeOperand>,
}

impl<'a> EdgeSelection<'a> {
    pub fn new<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<EdgeOperand>),
    {
        let mut operand = Wrapper::<EdgeOperand>::new();

        query(&mut operand);

        Self { medrecord, operand }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a EdgeIndex> {
        self.operand.evaluate(self.medrecord)
    }

    pub fn collect<B: FromIterator<&'a EdgeIndex>>(self) -> B {
        FromIterator::from_iter(self.iter())
    }
}

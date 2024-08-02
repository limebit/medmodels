use super::EdgeOperandWrapper;
use crate::medrecord::{EdgeIndex, MedRecord};

pub struct EdgeSelection<'a> {
    medrecord: &'a MedRecord,
    operand: EdgeOperandWrapper,
}

impl<'a> EdgeSelection<'a> {
    pub fn new<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut EdgeOperandWrapper),
    {
        let mut operand = EdgeOperandWrapper::new();

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

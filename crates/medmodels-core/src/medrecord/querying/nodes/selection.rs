use super::NodeOperand;
use crate::{
    errors::MedRecordResult,
    medrecord::{querying::wrapper::Wrapper, MedRecord, NodeIndex},
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

    pub fn iter(self) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        self.operand.evaluate(self.medrecord, None)
    }

    pub fn collect<B>(self) -> MedRecordResult<B>
    where
        B: FromIterator<&'a NodeIndex>,
    {
        Ok(FromIterator::from_iter(self.iter()?))
    }
}

use super::{wrapper::Wrapper, DeepClone, ReadWriteOrPanic};
use crate::{
    medrecord::querying::GroupedIterator,
    prelude::{MedRecordValue, NodeIndex},
    MedRecord,
};
use std::fmt::Debug;

pub trait GroupedOperand {
    type Context: Debug + Clone + DeepClone;
}

pub trait GroupBy: GroupedOperand {
    type Discriminator: Debug + Clone + DeepClone;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>>
    where
        Self: Sized;
}

impl<O: GroupBy> Wrapper<O> {
    pub fn group_by(&self, discriminator: O::Discriminator) -> Wrapper<GroupOperand<O>> {
        self.0.write_or_panic().group_by(discriminator)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum GroupKey<'a> {
    NodeIndex(&'a NodeIndex),
    Value(&'a MedRecordValue),
    OptionalValue(Option<&'a MedRecordValue>),
    TupleKey((Box<GroupKey<'a>>, Box<GroupKey<'a>>)),
}

pub trait PartitionGroups<'a>: GroupBy {
    type Values;

    fn partition(
        medrecord: &'a MedRecord,
        values: Self::Values,
        discriminator: &Self::Discriminator,
    ) -> GroupedIterator<'a, Self::Values>;

    fn merge(values: GroupedIterator<'a, Self::Values>) -> Self::Values;
}

pub trait Ungroup {
    type OutputOperand;

    fn ungroup(&self) -> Wrapper<Self::OutputOperand>;
}

impl<O: Ungroup> Wrapper<O> {
    pub fn ungroup(&self) -> Wrapper<O::OutputOperand> {
        self.0.read_or_panic().ungroup()
    }
}

#[derive(Debug, Clone)]
pub struct GroupOperand<O: GroupedOperand> {
    pub(crate) context: O::Context,
    pub(crate) operand: Wrapper<O>,
}

impl<O: GroupedOperand + DeepClone> DeepClone for GroupOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            operand: self.operand.deep_clone(),
        }
    }
}

impl<O: GroupedOperand> GroupOperand<O> {
    pub(crate) fn new(context: O::Context, operand: Wrapper<O>) -> Self {
        Self { context, operand }
    }
}

impl<O: GroupedOperand> Wrapper<GroupOperand<O>> {
    pub(crate) fn new(context: O::Context, operand: Wrapper<O>) -> Self {
        GroupOperand::new(context, operand).into()
    }
}

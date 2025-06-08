use super::{wrapper::Wrapper, DeepClone, ReadWriteOrPanic};
use crate::{
    medrecord::querying::GroupedIterator,
    prelude::{EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex},
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
    EdgeIndex(&'a EdgeIndex),
    Attribute(&'a MedRecordAttribute),
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

pub trait Merge {
    type OutputOperand;

    fn merge(&self) -> Wrapper<Self::OutputOperand>;
}

impl<O: Merge> Merge for Wrapper<O> {
    type OutputOperand = O::OutputOperand;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        self.0.read_or_panic().merge()
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

#[cfg(test)]
mod tests {
    use crate::{
        medrecord::querying::{
            edges::EdgeOperandGroupDiscriminator, group_by::Merge, nodes::EdgeDirection,
        },
        prelude::MedRecordAttribute,
        MedRecord,
    };

    #[test]
    fn test_group_by() {
        let medrecord = MedRecord::from_admissions_example_dataset();

        let result = medrecord
            .query_nodes(|nodes| {
                nodes.in_group(MedRecordAttribute::from("patient"));

                let edges = nodes.edges(EdgeDirection::Outgoing);

                edges
                    .target_node()
                    .in_group(MedRecordAttribute::from("admission"));

                let grouped_edges = edges.group_by(EdgeOperandGroupDiscriminator::SourceNode);

                grouped_edges.index().count().merge().is_max();

                (edges.index().count(), nodes.index())
            })
            .evaluate()
            .unwrap();

        let result = (result.0, result.1.collect::<Vec<_>>());

        println!("{:?} {:?}", result.0, result.1);
    }
}

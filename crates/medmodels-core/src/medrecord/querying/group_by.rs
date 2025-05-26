use super::{
    operand_traits::{Attribute, Max},
    wrapper::Wrapper,
    BoxedIterator, DeepClone, ReadWriteOrPanic,
};
use crate::{prelude::MedRecordAttribute, MedRecord};
use std::fmt::Debug;

pub trait GroupedOperand {
    type Context: Debug + Clone + DeepClone;
}

pub trait GroupableOperand: GroupedOperand {
    type Discriminator: Debug + Clone + DeepClone;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>>
    where
        Self: Sized;
}

impl<O: GroupableOperand> Wrapper<O> {
    pub fn group_by(&self, discriminator: O::Discriminator) -> Wrapper<GroupOperand<O>> {
        self.0.write_or_panic().group_by(discriminator)
    }
}

pub trait PartitionGroups<'a>: GroupableOperand {
    type GroupKey;
    type Values;

    fn partition(
        medrecord: &'a MedRecord,
        values: Self::Values,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (Self::GroupKey, Self::Values)>;

    fn merge(values: BoxedIterator<'a, (Self::GroupKey, Self::Values)>) -> Self::Values;
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

impl<O: GroupedOperand + Attribute> Attribute for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attribute(attribute);

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

impl<O: GroupedOperand + Max> Max for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.max();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
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

// #[cfg(test)]
// mod tests {
//     use crate::{
//         medrecord::querying::{
//             edges::EdgeOperandGroupDiscriminator, group_by::Merge, nodes::EdgeDirection,
//         },
//         prelude::MedRecordAttribute,
//         MedRecord,
//     };

//     #[test]
//     fn test_group_by() {
//         let medrecord = MedRecord::from_admissions_example_dataset();

//         let result = medrecord
//             .query_nodes(|nodes| {
//                 let mut edges = nodes.edges(EdgeDirection::Outgoing);

//                 edges.has_attribute(MedRecordAttribute::from("duration_days"));

//                 let group_by = edges.group_by(EdgeOperandGroupDiscriminator::SourceNode);

//                 group_by.attribute("duration_days").max().merge().max()
//             })
//             .evaluate()
//             .unwrap();

//         println!("{:?}", result);
//     }
// }

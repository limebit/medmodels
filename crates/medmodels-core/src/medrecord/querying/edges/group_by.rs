use super::EdgeOperand;
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        group_by::{GroupBy, GroupOperand, GroupedOperand, PartitionGroups},
        BoxedIterator, DeepClone, EvaluateForward, EvaluateForwardGrouped,
    },
    prelude::MedRecordAttribute,
    MedRecord,
};

#[derive(Debug, Clone)]
pub enum EdgeOperandContext {
    Discriminator(<EdgeOperand as GroupBy>::Discriminator),
}

impl DeepClone for EdgeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeOperandContext::Discriminator(discriminator) => {
                EdgeOperandContext::Discriminator(discriminator.clone())
            }
        }
    }
}

impl From<<EdgeOperand as GroupBy>::Discriminator> for EdgeOperandContext {
    fn from(discriminator: <EdgeOperand as GroupBy>::Discriminator) -> Self {
        EdgeOperandContext::Discriminator(discriminator)
    }
}

impl GroupedOperand for EdgeOperand {
    type Context = EdgeOperandContext;
}

#[derive(Debug, Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode,
    TargetNode,
    Parallel,
    Attribute(MedRecordAttribute),
}

impl DeepClone for EdgeOperandGroupDiscriminator {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeOperandGroupDiscriminator::SourceNode => EdgeOperandGroupDiscriminator::SourceNode,
            EdgeOperandGroupDiscriminator::TargetNode => EdgeOperandGroupDiscriminator::TargetNode,
            EdgeOperandGroupDiscriminator::Parallel => EdgeOperandGroupDiscriminator::Parallel,
            EdgeOperandGroupDiscriminator::Attribute(attr) => {
                EdgeOperandGroupDiscriminator::Attribute(attr.clone())
            }
        }
    }
}

impl<'a> EvaluateForward<'a> for GroupOperand<EdgeOperand> {
    type InputValue = <EdgeOperand as EvaluateForward<'a>>::InputValue;
    // type ReturnValue = BoxedIterator<
    //     'a,
    //     (
    //         <EdgeOperand as PartitionGroups<'a>>::GroupKey,
    //         <EdgeOperand as EvaluateForward<'a>>::ReturnValue,
    //     ),
    // >;
    type ReturnValue = BoxedIterator<'a, <EdgeOperand as EvaluateForward<'a>>::ReturnValue>;
    // type ReturnValue = <EdgeOperand as EvaluateForward<'a>>::ReturnValue;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let EdgeOperandContext::Discriminator(discriminator) = &self.context;

        let partitions = EdgeOperand::partition(medrecord, indices, discriminator);

        let indices = self.operand.evaluate_forward_grouped(
            medrecord,
            Box::new(partitions.map(|(_, parition)| parition)),
        )?;

        // let indices: Vec<_> = partitions
        //     .map(|(key, partition)| Ok((key, self.operand.evaluate_forward(medrecord, partition)?)))
        //     .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

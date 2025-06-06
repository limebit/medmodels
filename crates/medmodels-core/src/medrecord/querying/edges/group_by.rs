use super::EdgeOperand;
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeIndicesOperandContext},
        group_by::{GroupBy, GroupOperand, GroupedOperand, Merge, PartitionGroups},
        wrapper::Wrapper,
        BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
        GroupedIterator,
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
    type ReturnValue = GroupedIterator<'a, <EdgeOperand as EvaluateForward<'a>>::ReturnValue>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let EdgeOperandContext::Discriminator(discriminator) = &self.context;

        let partitions = EdgeOperand::partition(medrecord, indices, discriminator);

        self.operand
            .evaluate_forward_grouped(medrecord, Box::new(partitions))
    }
}

impl GroupedOperand for EdgeIndicesOperand {
    type Context = GroupOperand<EdgeOperand>;
}

impl<'a> EvaluateBackward<'a> for GroupOperand<EdgeIndicesOperand> {
    type ReturnValue = BoxedIterator<'a, <EdgeIndicesOperand as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let partitions = self.context.evaluate_backward(medrecord)?;

        let indices: Vec<_> = partitions
            .map(|partition| {
                self.operand
                    .evaluate_forward(medrecord, Box::new(partition.cloned()))
            })
            .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

impl GroupedOperand for EdgeIndexOperand {
    type Context = GroupOperand<EdgeIndicesOperand>;
}

impl<'a> EvaluateBackward<'a> for GroupOperand<EdgeIndexOperand> {
    type ReturnValue = BoxedIterator<'a, <EdgeIndexOperand as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let partitions = self.context.evaluate_backward(medrecord)?;

        let indices: Vec<_> = partitions
            .map(|partition| {
                let reduced_partition = self.operand.reduce_input(partition)?;

                self.operand.evaluate_forward(medrecord, reduced_partition)
            })
            .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

impl Merge for GroupOperand<EdgeIndexOperand> {
    type OutputOperand = EdgeIndicesOperand;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(EdgeIndicesOperandContext::GroupBy(
            self.deep_clone(),
        ));

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

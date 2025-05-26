use super::{MultipleValuesOperand, SingleValueOperand};
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        edges::EdgeOperand,
        group_by::{GroupOperand, GroupedOperand, Merge, PartitionGroups},
        values::Context,
        wrapper::Wrapper,
        BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic, RootOperand,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum MultipleValuesOperandContext<O: RootOperand> {
    RootOperand(GroupOperand<O>),
}

impl<O: RootOperand> DeepClone for MultipleValuesOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            MultipleValuesOperandContext::RootOperand(operand) => {
                MultipleValuesOperandContext::RootOperand(operand.deep_clone())
            }
        }
    }
}

impl From<GroupOperand<EdgeOperand>> for MultipleValuesOperandContext<EdgeOperand> {
    fn from(operand: GroupOperand<EdgeOperand>) -> Self {
        MultipleValuesOperandContext::RootOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for MultipleValuesOperand<O> {
    type Context = MultipleValuesOperandContext<O>;
}

impl<'a, O: RootOperand> EvaluateBackward<'a> for GroupOperand<MultipleValuesOperand<O>>
where
    O: 'a
        + PartitionGroups<'a, Values = <O as EvaluateBackward<'a>>::ReturnValue>
        + EvaluateBackward<'a, ReturnValue = BoxedIterator<'a, &'a O::Index>>,
    GroupOperand<O>: EvaluateBackward<
        'a,
        ReturnValue = BoxedIterator<'a, <O as EvaluateBackward<'a>>::ReturnValue>,
    >,
{
    type ReturnValue =
        BoxedIterator<'a, <MultipleValuesOperand<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            MultipleValuesOperandContext::RootOperand(context) => {
                let partitions = context.evaluate_backward(medrecord)?;

                let values: Vec<_> = partitions
                    .map(|partition| {
                        let Context::Operand((_, attribute)) =
                            &self.operand.0.read_or_panic().context
                        else {
                            unreachable!()
                        };

                        let reduced_partition =
                            O::get_values_from_indices(medrecord, attribute.clone(), partition);

                        self.operand
                            .evaluate_forward(medrecord, Box::new(reduced_partition))
                    })
                    .collect::<MedRecordResult<_>>()?;

                Ok(Box::new(values.into_iter()))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueOperandContext<O: RootOperand> {
    MultipleValuesOperand(GroupOperand<MultipleValuesOperand<O>>),
}

impl<O: RootOperand> DeepClone for SingleValueOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            SingleValueOperandContext::MultipleValuesOperand(operand) => {
                SingleValueOperandContext::MultipleValuesOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleValuesOperand<O>>> for SingleValueOperandContext<O> {
    fn from(operand: GroupOperand<MultipleValuesOperand<O>>) -> Self {
        SingleValueOperandContext::MultipleValuesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for SingleValueOperand<O> {
    type Context = SingleValueOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for GroupOperand<SingleValueOperand<O>>
where
    O: 'a
        + PartitionGroups<'a, Values = <O as EvaluateBackward<'a>>::ReturnValue>
        + EvaluateBackward<'a, ReturnValue = BoxedIterator<'a, &'a O::Index>>,
    GroupOperand<O>: EvaluateBackward<
        'a,
        ReturnValue = BoxedIterator<'a, <O as EvaluateBackward<'a>>::ReturnValue>,
    >,
{
    type ReturnValue =
        BoxedIterator<'a, <SingleValueOperand<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            SingleValueOperandContext::MultipleValuesOperand(context) => {
                let partitions = context.evaluate_backward(medrecord)?;

                let values: Vec<_> = partitions
                    .map(|partition| {
                        let reduced_partition = self.operand.reduce_input(partition)?;

                        self.operand.evaluate_forward(medrecord, reduced_partition)
                    })
                    .collect::<MedRecordResult<_>>()?;

                Ok(Box::new(values.into_iter()))
            }
        }
    }
}

impl<O: RootOperand> Merge for GroupOperand<SingleValueOperand<O>> {
    type OutputOperand = MultipleValuesOperand<O>;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        Wrapper::<Self::OutputOperand>::new(Context::GroupByOperand(self.deep_clone()))
    }
}

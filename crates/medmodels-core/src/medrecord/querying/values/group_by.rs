use super::{MultipleValuesOperandWithIndex, SingleValueOperandWithIndex};
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        edges::EdgeOperand,
        group_by::{GroupOperand, GroupedOperand, Merge},
        values::{
            operand::MultipleValuesOperandWithoutIndex,
            operation::{MultipleValuesOperationWithIndex, MultipleValuesOperationWithoutIndex},
            MultipleValuesWithIndexContext, MultipleValuesWithoutIndexContext,
            SingleKindWithoutIndex, SingleValueOperandWithoutIndex,
        },
        wrapper::Wrapper,
        BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic, RootOperand,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum MultipleValuesOperandWithIndexContext<O: RootOperand> {
    RootOperand(GroupOperand<O>),
}

impl<O: RootOperand> DeepClone for MultipleValuesOperandWithIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            MultipleValuesOperandWithIndexContext::RootOperand(operand) => {
                MultipleValuesOperandWithIndexContext::RootOperand(operand.deep_clone())
            }
        }
    }
}

impl From<GroupOperand<EdgeOperand>> for MultipleValuesOperandWithIndexContext<EdgeOperand> {
    fn from(operand: GroupOperand<EdgeOperand>) -> Self {
        MultipleValuesOperandWithIndexContext::RootOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for MultipleValuesOperandWithIndex<O> {
    type Context = MultipleValuesOperandWithIndexContext<O>;
}

impl<'a, O: RootOperand> EvaluateBackward<'a> for GroupOperand<MultipleValuesOperandWithIndex<O>>
where
    O: 'a,
{
    type ReturnValue =
        BoxedIterator<'a, <MultipleValuesOperandWithIndex<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            MultipleValuesOperandWithIndexContext::RootOperand(context) => {
                let partitions = context.evaluate_backward(medrecord)?;

                let values: Vec<_> = partitions
                    .map(|partition| {
                        let MultipleValuesWithIndexContext::Operand((_, attribute)) =
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
pub enum SingleValueOperandWithIndexContext<O: RootOperand> {
    MultipleValuesOperand(GroupOperand<MultipleValuesOperandWithIndex<O>>),
}

impl<O: RootOperand> DeepClone for SingleValueOperandWithIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            SingleValueOperandWithIndexContext::MultipleValuesOperand(operand) => {
                SingleValueOperandWithIndexContext::MultipleValuesOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleValuesOperandWithIndex<O>>>
    for SingleValueOperandWithIndexContext<O>
{
    fn from(operand: GroupOperand<MultipleValuesOperandWithIndex<O>>) -> Self {
        SingleValueOperandWithIndexContext::MultipleValuesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for SingleValueOperandWithIndex<O> {
    type Context = SingleValueOperandWithIndexContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleValueOperandWithIndex<O>>
{
    type ReturnValue =
        BoxedIterator<'a, <SingleValueOperandWithIndex<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            SingleValueOperandWithIndexContext::MultipleValuesOperand(context) => {
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

impl<O: RootOperand> Merge for GroupOperand<SingleValueOperandWithIndex<O>> {
    type OutputOperand = MultipleValuesOperandWithIndex<O>;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            MultipleValuesWithIndexContext::GroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueOperandWithoutIndexContext<O: RootOperand> {
    MultipleValuesOperand(GroupOperand<MultipleValuesOperandWithIndex<O>>),
}

impl<O: RootOperand> DeepClone for SingleValueOperandWithoutIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::MultipleValuesOperand(operand) => {
                Self::MultipleValuesOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleValuesOperandWithIndex<O>>>
    for SingleValueOperandWithoutIndexContext<O>
{
    fn from(operand: GroupOperand<MultipleValuesOperandWithIndex<O>>) -> Self {
        Self::MultipleValuesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for SingleValueOperandWithoutIndex<O> {
    type Context = SingleValueOperandWithoutIndexContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleValueOperandWithoutIndex<O>>
{
    type ReturnValue =
        BoxedIterator<'a, <SingleValueOperandWithoutIndex<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            SingleValueOperandWithoutIndexContext::MultipleValuesOperand(context) => {
                let partitions = context.evaluate_backward(medrecord)?;

                let values: Vec<_> = partitions
                    .map(|partition| {
                        let partition = partition.map(|(_, value)| value);

                        let reduced_partition = match self.operand.0.read_or_panic().kind {
                            SingleKindWithoutIndex::Max => {
                                MultipleValuesOperationWithoutIndex::<O>::get_max(partition)?
                            }
                            SingleKindWithoutIndex::Min => {
                                MultipleValuesOperationWithoutIndex::<O>::get_min(partition)?
                            }
                            SingleKindWithoutIndex::Mean => {
                                MultipleValuesOperationWithIndex::<O>::get_mean(partition)?
                            }
                            SingleKindWithoutIndex::Median => {
                                MultipleValuesOperationWithIndex::<O>::get_median(partition)?
                            }
                            SingleKindWithoutIndex::Mode => {
                                MultipleValuesOperationWithIndex::<O>::get_mode(partition)?
                            }
                            SingleKindWithoutIndex::Std => {
                                MultipleValuesOperationWithIndex::<O>::get_std(partition)?
                            }
                            SingleKindWithoutIndex::Var => {
                                MultipleValuesOperationWithIndex::<O>::get_var(partition)?
                            }
                            SingleKindWithoutIndex::Count => {
                                MultipleValuesOperationWithIndex::<O>::get_count(partition)
                            }
                            SingleKindWithoutIndex::Sum => {
                                MultipleValuesOperationWithIndex::<O>::get_sum(partition)?
                            }
                            SingleKindWithoutIndex::Random => {
                                MultipleValuesOperationWithoutIndex::<O>::get_random(partition)?
                            }
                        };

                        self.operand.evaluate_forward(medrecord, reduced_partition)
                    })
                    .collect::<MedRecordResult<_>>()?;

                Ok(Box::new(values.into_iter()))
            }
        }
    }
}

impl<O: RootOperand> Merge for GroupOperand<SingleValueOperandWithoutIndex<O>> {
    type OutputOperand = MultipleValuesOperandWithoutIndex<O>;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            MultipleValuesWithoutIndexContext::GroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

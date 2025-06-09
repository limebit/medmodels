use super::{MultipleValuesWithIndexOperand, SingleValueWithIndexOperand};
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        attributes::{MultipleAttributesOperationWithIndex, MultipleAttributesWithIndexOperand},
        group_by::{GroupOperand, GroupedOperand, Merge},
        values::{
            operand::MultipleValuesWithoutIndexOperand,
            operation::MultipleValuesOperationWithoutIndex, MultipleValuesWithIndexContext,
            MultipleValuesWithoutIndexContext, SingleKindWithoutIndex,
            SingleValueWithoutIndexOperand,
        },
        wrapper::Wrapper,
        BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic, RootOperand,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum MultipleValuesWithIndexOperandContext<O: RootOperand> {
    RootOperand(GroupOperand<O>),
    MultipleAttributesOperand(GroupOperand<MultipleAttributesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for MultipleValuesWithIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            MultipleValuesWithIndexOperandContext::RootOperand(operand) => {
                MultipleValuesWithIndexOperandContext::RootOperand(operand.deep_clone())
            }
            MultipleValuesWithIndexOperandContext::MultipleAttributesOperand(operand) => {
                MultipleValuesWithIndexOperandContext::MultipleAttributesOperand(
                    operand.deep_clone(),
                )
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<O>> for MultipleValuesWithIndexOperandContext<O> {
    fn from(operand: GroupOperand<O>) -> Self {
        Self::RootOperand(operand)
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleAttributesWithIndexOperand<O>>>
    for MultipleValuesWithIndexOperandContext<O>
{
    fn from(operand: GroupOperand<MultipleAttributesWithIndexOperand<O>>) -> Self {
        Self::MultipleAttributesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for MultipleValuesWithIndexOperand<O> {
    type Context = MultipleValuesWithIndexOperandContext<O>;
}

impl<'a, O: RootOperand> EvaluateBackward<'a> for GroupOperand<MultipleValuesWithIndexOperand<O>>
where
    O: 'a,
{
    type ReturnValue =
        BoxedIterator<'a, <MultipleValuesWithIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            MultipleValuesWithIndexOperandContext::RootOperand(context) => {
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
            MultipleValuesWithIndexOperandContext::MultipleAttributesOperand(context) => {
                let partitions = context.evaluate_backward(medrecord)?;

                let values: Vec<_> = partitions
                    .map(|partition| {
                        let reduced_partition =
                            MultipleAttributesOperationWithIndex::<O>::get_values(
                                medrecord, partition,
                            )?;

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
pub enum SingleValueWithIndexOperandContext<O: RootOperand> {
    MultipleValuesOperand(GroupOperand<MultipleValuesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for SingleValueWithIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            SingleValueWithIndexOperandContext::MultipleValuesOperand(operand) => {
                SingleValueWithIndexOperandContext::MultipleValuesOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleValuesWithIndexOperand<O>>>
    for SingleValueWithIndexOperandContext<O>
{
    fn from(operand: GroupOperand<MultipleValuesWithIndexOperand<O>>) -> Self {
        SingleValueWithIndexOperandContext::MultipleValuesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for SingleValueWithIndexOperand<O> {
    type Context = SingleValueWithIndexOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleValueWithIndexOperand<O>>
{
    type ReturnValue =
        BoxedIterator<'a, <SingleValueWithIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            SingleValueWithIndexOperandContext::MultipleValuesOperand(context) => {
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

impl<O: RootOperand> Merge for GroupOperand<SingleValueWithIndexOperand<O>> {
    type OutputOperand = MultipleValuesWithIndexOperand<O>;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            MultipleValuesWithIndexContext::GroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueWithoutIndexOperandContext<O: RootOperand> {
    MultipleValuesOperand(GroupOperand<MultipleValuesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for SingleValueWithoutIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::MultipleValuesOperand(operand) => {
                Self::MultipleValuesOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> From<GroupOperand<MultipleValuesWithIndexOperand<O>>>
    for SingleValueWithoutIndexOperandContext<O>
{
    fn from(operand: GroupOperand<MultipleValuesWithIndexOperand<O>>) -> Self {
        Self::MultipleValuesOperand(operand)
    }
}

impl<O: RootOperand> GroupedOperand for SingleValueWithoutIndexOperand<O> {
    type Context = SingleValueWithoutIndexOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleValueWithoutIndexOperand<O>>
{
    type ReturnValue =
        BoxedIterator<'a, <SingleValueWithoutIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            SingleValueWithoutIndexOperandContext::MultipleValuesOperand(context) => {
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
                                MultipleValuesOperationWithoutIndex::<O>::get_mean(partition)?
                            }
                            SingleKindWithoutIndex::Median => {
                                MultipleValuesOperationWithoutIndex::<O>::get_median(partition)?
                            }
                            SingleKindWithoutIndex::Mode => {
                                MultipleValuesOperationWithoutIndex::<O>::get_mode(partition)?
                            }
                            SingleKindWithoutIndex::Std => {
                                MultipleValuesOperationWithoutIndex::<O>::get_std(partition)?
                            }
                            SingleKindWithoutIndex::Var => {
                                MultipleValuesOperationWithoutIndex::<O>::get_var(partition)?
                            }
                            SingleKindWithoutIndex::Count => Some(
                                MultipleValuesOperationWithoutIndex::<O>::get_count(partition),
                            ),
                            SingleKindWithoutIndex::Sum => {
                                MultipleValuesOperationWithoutIndex::<O>::get_sum(partition)?
                            }
                            SingleKindWithoutIndex::Random => {
                                MultipleValuesOperationWithoutIndex::<O>::get_random(partition)
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

impl<O: RootOperand> Merge for GroupOperand<SingleValueWithoutIndexOperand<O>> {
    type OutputOperand = MultipleValuesWithoutIndexOperand<O>;

    fn merge(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            MultipleValuesWithoutIndexContext::GroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

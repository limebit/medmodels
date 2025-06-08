use crate::medrecord::querying::{
    attributes::{
        MultipleAttributesOperandWithIndex, SingleAttributeOperandWithIndex,
        SingleAttributeOperandWithoutIndex,
    },
    group_by::{GroupOperand, GroupedOperand},
    BoxedIterator, DeepClone, EvaluateBackward, RootOperand,
};

#[derive(Debug, Clone)]
pub enum MultipleAttributesOperandWithIndexContext<O: RootOperand> {
    RootOperand(GroupOperand<O>),
}

impl<O: RootOperand> DeepClone for MultipleAttributesOperandWithIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            MultipleAttributesOperandWithIndexContext::RootOperand(operand) => {
                MultipleAttributesOperandWithIndexContext::RootOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> GroupedOperand for MultipleAttributesOperandWithIndex<O> {
    type Context = MultipleAttributesOperandWithIndexContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<MultipleAttributesOperandWithIndex<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <MultipleAttributesOperandWithIndex<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

// #[derive(Debug, Clone)]
// pub enum MultipleAttributesOperandWithoutIndexContext {}

// impl DeepClone for MultipleAttributesOperandWithoutIndexContext {
//     fn deep_clone(&self) -> Self {
//         self.clone()
//     }
// }

// impl<O: RootOperand> GroupedOperand for MultipleAttributesOperandWithoutIndex<O> {
//     type Context = MultipleAttributesOperandWithoutIndexContext;
// }

#[derive(Debug, Clone)]
pub enum SingleAttributeOperandWithIndexContext<O: RootOperand> {
    MultipleAttributesOperand(GroupOperand<MultipleAttributesOperandWithIndex<O>>),
}

impl<O: RootOperand> DeepClone for SingleAttributeOperandWithIndexContext<O> {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl<O: RootOperand> GroupedOperand for SingleAttributeOperandWithIndex<O> {
    type Context = SingleAttributeOperandWithIndexContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleAttributeOperandWithIndex<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <SingleAttributeOperandWithIndex<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeOperandWithoutIndexContext<O: RootOperand> {
    MultipleAttributesOperand(GroupOperand<MultipleAttributesOperandWithIndex<O>>),
}

impl<O: RootOperand> DeepClone for SingleAttributeOperandWithoutIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            SingleAttributeOperandWithoutIndexContext::MultipleAttributesOperand(operand) => {
                SingleAttributeOperandWithoutIndexContext::MultipleAttributesOperand(
                    operand.deep_clone(),
                )
            }
        }
    }
}

impl<O: RootOperand> GroupedOperand for SingleAttributeOperandWithoutIndex<O> {
    type Context = SingleAttributeOperandWithoutIndexContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleAttributeOperandWithoutIndex<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <SingleAttributeOperandWithoutIndex<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

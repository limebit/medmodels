use crate::medrecord::querying::{
    attributes::{
        MultipleAttributesWithIndexOperand, SingleAttributeWithIndexOperand,
        SingleAttributeWithoutIndexOperand,
    },
    group_by::{GroupOperand, GroupedOperand},
    BoxedIterator, DeepClone, EvaluateBackward, RootOperand,
};

#[derive(Debug, Clone)]
pub enum MultipleAttributesWithIndexOperandContext<O: RootOperand> {
    RootOperand(GroupOperand<O>),
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            MultipleAttributesWithIndexOperandContext::RootOperand(operand) => {
                MultipleAttributesWithIndexOperandContext::RootOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> GroupedOperand for MultipleAttributesWithIndexOperand<O> {
    type Context = MultipleAttributesWithIndexOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<MultipleAttributesWithIndexOperand<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <MultipleAttributesWithIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

// #[derive(Debug, Clone)]
// pub enum MultipleAttributesWithoutIndexOperandContext {}

// impl DeepClone for MultipleAttributesWithoutIndexOperandContext {
//     fn deep_clone(&self) -> Self {
//         self.clone()
//     }
// }

// impl<O: RootOperand> GroupedOperand for MultipleAttributesWithoutIndexOperand<O> {
//     type Context = MultipleAttributesWithoutIndexOperandContext;
// }

#[derive(Debug, Clone)]
pub enum SingleAttributeWithIndexOperandContext<O: RootOperand> {
    MultipleAttributesOperand(GroupOperand<MultipleAttributesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for SingleAttributeWithIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl<O: RootOperand> GroupedOperand for SingleAttributeWithIndexOperand<O> {
    type Context = SingleAttributeWithIndexOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleAttributeWithIndexOperand<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <SingleAttributeWithIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeWithoutIndexOperandContext<O: RootOperand> {
    MultipleAttributesOperand(GroupOperand<MultipleAttributesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for SingleAttributeWithoutIndexOperandContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            SingleAttributeWithoutIndexOperandContext::MultipleAttributesOperand(operand) => {
                SingleAttributeWithoutIndexOperandContext::MultipleAttributesOperand(
                    operand.deep_clone(),
                )
            }
        }
    }
}

impl<O: RootOperand> GroupedOperand for SingleAttributeWithoutIndexOperand<O> {
    type Context = SingleAttributeWithoutIndexOperandContext<O>;
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a>
    for GroupOperand<SingleAttributeWithoutIndexOperand<O>>
{
    type ReturnValue = BoxedIterator<
        'a,
        <SingleAttributeWithoutIndexOperand<O> as EvaluateBackward<'a>>::ReturnValue,
    >;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> crate::errors::MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

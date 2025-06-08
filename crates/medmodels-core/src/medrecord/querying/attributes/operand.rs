use super::{
    operation::{
        AttributesTreeOperation, MultipleAttributesOperationWithIndex,
        SingleAttributeOperationWithIndex,
    },
    BinaryArithmeticKind, MultipleComparisonKind, MultipleKind, SingleComparisonKind,
    SingleKindWithIndex, UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::{
                operation::{
                    MultipleAttributesOperationWithoutIndex, SingleAttributeOperationWithoutIndex,
                },
                MultipleAttributesWithIndexContext, MultipleAttributesWithoutIndexContext,
                SingleAttributeWithoutIndexContext, SingleKindWithoutIndex,
            },
            values::{MultipleValuesOperandWithIndex, MultipleValuesWithIndexContext},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeOperand, MedRecordAttribute, NodeOperand, Wrapper,
    },
    MedRecord,
};
use medmodels_utils::aliases::MrHashSet;
use std::collections::HashSet;

macro_rules! implement_attributes_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<MultipleAttributesOperandWithIndex<O>> {
            let operand = Wrapper::<MultipleAttributesOperandWithIndex<O>>::new(
                MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
                MultipleKind::$variant,
            );

            self.operations
                .push(AttributesTreeOperation::AttributesOperation {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_attribute_operation_with_index {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleAttributeOperandWithIndex<O>> {
            let operand = Wrapper::<SingleAttributeOperandWithIndex<O>>::new(
                self.deep_clone(),
                SingleKindWithIndex::$variant,
            );

            self.operations
                .push(MultipleAttributesOperationWithIndex::AttributeOperation {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_attribute_operation_without_index {
    ($name:ident, $variant:ident, WithIndex) => {
        pub fn $name(&mut self) -> Wrapper<SingleAttributeOperandWithoutIndex<O>> {
            let operand = Wrapper::<SingleAttributeOperandWithoutIndex<O>>::new(
                SingleAttributeWithoutIndexContext::MultipleAttributesOperandWithIndex(
                    self.deep_clone(),
                ),
                SingleKindWithoutIndex::$variant,
            );

            self.operations.push(
                MultipleAttributesOperationWithIndex::AttributeOperationWithoutIndex {
                    operand: operand.clone(),
                },
            );

            operand
        }
    };
    ($name:ident, $variant:ident, WithoutIndex) => {
        pub fn $name(&mut self) -> Wrapper<SingleAttributeOperandWithoutIndex<O>> {
            let operand = Wrapper::<SingleAttributeOperandWithoutIndex<O>>::new(
                SingleAttributeWithoutIndexContext::MultipleAttributesOperandWithoutIndex(
                    self.deep_clone(),
                ),
                SingleKindWithoutIndex::$variant,
            );

            self.operations.push(
                MultipleAttributesOperationWithoutIndex::AttributeOperation {
                    operand: operand.clone(),
                },
            );

            operand
        }
    };
}

macro_rules! implement_single_attribute_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<SingleAttributeComparisonOperand>>(&mut self, attribute: V) {
            self.operations
                .push($operation::SingleAttributeComparisonOperation {
                    operand: attribute.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<SingleAttributeComparisonOperand>>(&mut self, attribute: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: attribute.into(),
                kind: BinaryArithmeticKind::$kind,
            });
        }
    };
}

macro_rules! implement_unary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name(&mut self) {
            self.operations.push($operation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::$kind,
            });
        }
    };
}

macro_rules! implement_assertion_operation {
    ($name:ident, $operation:expr) => {
        pub fn $name(&mut self) {
            self.operations.push($operation);
        }
    };
}

macro_rules! implement_wrapper_operand {
    ($name:ident) => {
        pub fn $name(&self) {
            self.0.write_or_panic().$name()
        }
    };
}

macro_rules! implement_wrapper_operand_with_return {
    ($name:ident, $return_operand:ty) => {
        pub fn $name(&self) -> Wrapper<$return_operand> {
            self.0.write_or_panic().$name()
        }
    };
}

macro_rules! implement_wrapper_operand_with_argument {
    ($name:ident, $attribute_type:ty) => {
        pub fn $name(&self, attribute: $attribute_type) {
            self.0.write_or_panic().$name(attribute)
        }
    };
}

#[derive(Debug, Clone)]
pub enum SingleAttributeComparisonOperand {
    NodeSingleAttributeOperandWithIndex(NodeSingleAttributeOperandWithIndex),
    NodeSingleAttributeOperandWithoutIndex(NodeSingleAttributeOperandWithoutIndex),
    EdgeSingleAttributeOperandWithIndex(EdgeSingleAttributeOperandWithIndex),
    EdgeSingleAttributeOperandWithoutIndex(EdgeSingleAttributeOperandWithoutIndex),
    Attribute(MedRecordAttribute),
}

impl DeepClone for SingleAttributeComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeSingleAttributeOperandWithIndex(operand) => {
                Self::NodeSingleAttributeOperandWithIndex(operand.deep_clone())
            }
            Self::NodeSingleAttributeOperandWithoutIndex(operand) => {
                Self::NodeSingleAttributeOperandWithoutIndex(operand.deep_clone())
            }
            Self::EdgeSingleAttributeOperandWithIndex(operand) => {
                Self::EdgeSingleAttributeOperandWithIndex(operand.deep_clone())
            }
            Self::EdgeSingleAttributeOperandWithoutIndex(operand) => {
                Self::EdgeSingleAttributeOperandWithoutIndex(operand.deep_clone())
            }
            Self::Attribute(attribute) => Self::Attribute(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeSingleAttributeOperandWithIndex>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<NodeSingleAttributeOperandWithIndex>) -> Self {
        Self::NodeSingleAttributeOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleAttributeOperandWithIndex>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<NodeSingleAttributeOperandWithIndex>) -> Self {
        Self::NodeSingleAttributeOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<NodeSingleAttributeOperandWithoutIndex>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<NodeSingleAttributeOperandWithoutIndex>) -> Self {
        Self::NodeSingleAttributeOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleAttributeOperandWithoutIndex>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<NodeSingleAttributeOperandWithoutIndex>) -> Self {
        Self::NodeSingleAttributeOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleAttributeOperandWithIndex>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<EdgeSingleAttributeOperandWithIndex>) -> Self {
        Self::EdgeSingleAttributeOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleAttributeOperandWithIndex>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleAttributeOperandWithIndex>) -> Self {
        Self::EdgeSingleAttributeOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleAttributeOperandWithoutIndex>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<EdgeSingleAttributeOperandWithoutIndex>) -> Self {
        Self::EdgeSingleAttributeOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleAttributeOperandWithoutIndex>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleAttributeOperandWithoutIndex>) -> Self {
        Self::EdgeSingleAttributeOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>> From<V> for SingleAttributeComparisonOperand {
    fn from(value: V) -> Self {
        Self::Attribute(value.into())
    }
}

impl SingleAttributeComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        Ok(match self {
            Self::NodeSingleAttributeOperandWithIndex(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::NodeSingleAttributeOperandWithoutIndex(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::EdgeSingleAttributeOperandWithIndex(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::EdgeSingleAttributeOperandWithoutIndex(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::Attribute(attribute) => Some(attribute.clone()),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesComparisonOperand {
    NodeMultipleAttributesOperandWithIndex(NodeMultipleAttributesOperandWithIndex),
    NodeMultipleAttributesOperandWithoutIndex(NodeMultipleAttributesOperandWithoutIndex),
    EdgeMultipleAttributesOperandWithIndex(EdgeMultipleAttributesOperandWithIndex),
    EdgeMultipleAttributesOperandWithoutIndex(EdgeMultipleAttributesOperandWithoutIndex),
    Attributes(MrHashSet<MedRecordAttribute>),
}

impl DeepClone for MultipleAttributesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeMultipleAttributesOperandWithIndex(operand) => {
                Self::NodeMultipleAttributesOperandWithIndex(operand.deep_clone())
            }
            Self::NodeMultipleAttributesOperandWithoutIndex(operand) => {
                Self::NodeMultipleAttributesOperandWithoutIndex(operand.deep_clone())
            }
            Self::EdgeMultipleAttributesOperandWithIndex(operand) => {
                Self::EdgeMultipleAttributesOperandWithIndex(operand.deep_clone())
            }
            Self::EdgeMultipleAttributesOperandWithoutIndex(operand) => {
                Self::EdgeMultipleAttributesOperandWithoutIndex(operand.deep_clone())
            }
            Self::Attributes(attribute) => Self::Attributes(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeMultipleAttributesOperandWithIndex>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleAttributesOperandWithIndex>) -> Self {
        Self::NodeMultipleAttributesOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleAttributesOperandWithIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<NodeMultipleAttributesOperandWithIndex>) -> Self {
        Self::NodeMultipleAttributesOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}
impl From<Wrapper<NodeMultipleAttributesOperandWithoutIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: Wrapper<NodeMultipleAttributesOperandWithoutIndex>) -> Self {
        Self::NodeMultipleAttributesOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleAttributesOperandWithoutIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<NodeMultipleAttributesOperandWithoutIndex>) -> Self {
        Self::NodeMultipleAttributesOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleAttributesOperandWithIndex>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleAttributesOperandWithIndex>) -> Self {
        Self::EdgeMultipleAttributesOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleAttributesOperandWithIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<EdgeMultipleAttributesOperandWithIndex>) -> Self {
        Self::EdgeMultipleAttributesOperandWithIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleAttributesOperandWithoutIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: Wrapper<EdgeMultipleAttributesOperandWithoutIndex>) -> Self {
        Self::EdgeMultipleAttributesOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleAttributesOperandWithoutIndex>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<EdgeMultipleAttributesOperandWithoutIndex>) -> Self {
        Self::EdgeMultipleAttributesOperandWithoutIndex(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>> From<Vec<V>> for MultipleAttributesComparisonOperand {
    fn from(value: Vec<V>) -> Self {
        Self::Attributes(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordAttribute>> From<HashSet<V>> for MultipleAttributesComparisonOperand {
    fn from(value: HashSet<V>) -> Self {
        Self::Attributes(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordAttribute>> From<MrHashSet<V>> for MultipleAttributesComparisonOperand {
    fn from(value: MrHashSet<V>) -> Self {
        Self::Attributes(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordAttribute> + Clone, const N: usize> From<[V; N]>
    for MultipleAttributesComparisonOperand
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

impl MultipleAttributesComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<MrHashSet<MedRecordAttribute>> {
        Ok(match self {
            Self::NodeMultipleAttributesOperandWithIndex(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::NodeMultipleAttributesOperandWithoutIndex(operand) => {
                operand.evaluate_backward(medrecord)?.collect()
            }
            Self::EdgeMultipleAttributesOperandWithIndex(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::EdgeMultipleAttributesOperandWithoutIndex(operand) => {
                operand.evaluate_backward(medrecord)?.collect()
            }
            Self::Attributes(attributes) => attributes.clone(),
        })
    }
}

pub type NodeAttributesTreeOperand = AttributesTreeOperand<NodeOperand>;
pub type EdgeAttributesTreeOperand = AttributesTreeOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct AttributesTreeOperand<O: RootOperand> {
    context: O,
    operations: Vec<AttributesTreeOperation<O>>,
}

impl<O: RootOperand> DeepClone for AttributesTreeOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for AttributesTreeOperand<O> {
    type InputValue = BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>;
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        attributes: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let attributes = Box::new(attributes) as BoxedIterator<_>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for AttributesTreeOperand<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate_grouped(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for AttributesTreeOperand<O> {
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes: BoxedIterator<_> = Box::new(self.context.get_attributes(medrecord)?);

        self.evaluate_forward(medrecord, attributes)
    }
}

impl<O: RootOperand> AttributesTreeOperand<O> {
    pub(crate) fn new(context: O) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    implement_attributes_operation!(max, Max);
    implement_attributes_operation!(min, Min);
    implement_attributes_operation!(count, Count);
    implement_attributes_operation!(sum, Sum);
    implement_attributes_operation!(random, Random);

    implement_single_attribute_comparison_operation!(
        greater_than,
        AttributesTreeOperation,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        AttributesTreeOperation,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(less_than, AttributesTreeOperation, LessThan);
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        AttributesTreeOperation,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(equal_to, AttributesTreeOperation, EqualTo);
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        AttributesTreeOperation,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        AttributesTreeOperation,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(ends_with, AttributesTreeOperation, EndsWith);
    implement_single_attribute_comparison_operation!(contains, AttributesTreeOperation, Contains);

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            AttributesTreeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            AttributesTreeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, AttributesTreeOperation, Add);
    implement_binary_arithmetic_operation!(sub, AttributesTreeOperation, Sub);
    implement_binary_arithmetic_operation!(mul, AttributesTreeOperation, Mul);
    implement_binary_arithmetic_operation!(pow, AttributesTreeOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, AttributesTreeOperation, Mod);

    implement_unary_arithmetic_operation!(abs, AttributesTreeOperation, Abs);
    implement_unary_arithmetic_operation!(trim, AttributesTreeOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, AttributesTreeOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, AttributesTreeOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, AttributesTreeOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, AttributesTreeOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(AttributesTreeOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, AttributesTreeOperation::IsString);
    implement_assertion_operation!(is_int, AttributesTreeOperation::IsInt);
    implement_assertion_operation!(is_max, AttributesTreeOperation::IsMax);
    implement_assertion_operation!(is_min, AttributesTreeOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
        OQ: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
    {
        let mut either_operand = Wrapper::<AttributesTreeOperand<O>>::new(self.context.clone());
        let mut or_operand = Wrapper::<AttributesTreeOperand<O>>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(AttributesTreeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
    {
        let mut operand = Wrapper::<AttributesTreeOperand<O>>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(AttributesTreeOperation::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<AttributesTreeOperand<O>> {
    pub(crate) fn new(context: O) -> Self {
        AttributesTreeOperand::new(context).into()
    }

    implement_wrapper_operand_with_return!(max, MultipleAttributesOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(min, MultipleAttributesOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(count, MultipleAttributesOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(sum, MultipleAttributesOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(random, MultipleAttributesOperandWithIndex<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleAttributesComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
        OQ: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<AttributesTreeOperand<O>>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

pub type NodeMultipleAttributesOperandWithIndex = MultipleAttributesOperandWithIndex<NodeOperand>;
pub type EdgeMultipleAttributesOperandWithIndex = MultipleAttributesOperandWithIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleAttributesOperandWithIndex<O: RootOperand> {
    context: MultipleAttributesWithIndexContext<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleAttributesOperandWithIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleAttributesOperandWithIndex<O> {
    type InputValue = BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>;
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        attributes: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let attributes = Box::new(attributes) as BoxedIterator<_>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for MultipleAttributesOperandWithIndex<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate_grouped(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleAttributesOperandWithIndex<O> {
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.get_attributes(medrecord, &self.kind)?;

        self.evaluate_forward(medrecord, attributes)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for MultipleAttributesOperandWithIndex<O> {
    type Context = AttributesTreeOperand<O>;

    #[inline]
    fn reduce_input(
        &self,
        attributes: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            MultipleKind::Max => Box::new(AttributesTreeOperation::<O>::get_max(attributes)?),
            MultipleKind::Min => Box::new(AttributesTreeOperation::<O>::get_min(attributes)?),
            MultipleKind::Count => Box::new(AttributesTreeOperation::<O>::get_count(attributes)?),
            MultipleKind::Sum => Box::new(AttributesTreeOperation::<O>::get_sum(attributes)?),
            MultipleKind::Random => Box::new(AttributesTreeOperation::<O>::get_random(attributes)?),
        })
    }
}

impl<O: RootOperand> MultipleAttributesOperandWithIndex<O> {
    pub(crate) fn new(context: MultipleAttributesWithIndexContext<O>, kind: MultipleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_attribute_operation_with_index!(max, Max);
    implement_attribute_operation_with_index!(min, Min);
    implement_attribute_operation_without_index!(count, Count, WithIndex);
    implement_attribute_operation_without_index!(sum, Sum, WithIndex);
    implement_attribute_operation_with_index!(random, Random);

    implement_single_attribute_comparison_operation!(
        greater_than,
        MultipleAttributesOperationWithIndex,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        MultipleAttributesOperationWithIndex,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        less_than,
        MultipleAttributesOperationWithIndex,
        LessThan
    );
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        MultipleAttributesOperationWithIndex,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        equal_to,
        MultipleAttributesOperationWithIndex,
        EqualTo
    );
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        MultipleAttributesOperationWithIndex,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        MultipleAttributesOperationWithIndex,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(
        ends_with,
        MultipleAttributesOperationWithIndex,
        EndsWith
    );
    implement_single_attribute_comparison_operation!(
        contains,
        MultipleAttributesOperationWithIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, MultipleAttributesOperationWithIndex, Add);
    implement_binary_arithmetic_operation!(sub, MultipleAttributesOperationWithIndex, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleAttributesOperationWithIndex, Mul);
    implement_binary_arithmetic_operation!(pow, MultipleAttributesOperationWithIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleAttributesOperationWithIndex, Mod);

    implement_unary_arithmetic_operation!(abs, MultipleAttributesOperationWithIndex, Abs);
    implement_unary_arithmetic_operation!(trim, MultipleAttributesOperationWithIndex, Trim);
    implement_unary_arithmetic_operation!(
        trim_start,
        MultipleAttributesOperationWithIndex,
        TrimStart
    );
    implement_unary_arithmetic_operation!(trim_end, MultipleAttributesOperationWithIndex, TrimEnd);
    implement_unary_arithmetic_operation!(
        lowercase,
        MultipleAttributesOperationWithIndex,
        Lowercase
    );
    implement_unary_arithmetic_operation!(
        uppercase,
        MultipleAttributesOperationWithIndex,
        Uppercase
    );

    #[allow(clippy::wrong_self_convention)]
    pub fn to_values(&mut self) -> Wrapper<MultipleValuesOperandWithIndex<O>> {
        let operand = Wrapper::<MultipleValuesOperandWithIndex<O>>::new(
            MultipleValuesWithIndexContext::MultipleAttributesOperand(self.deep_clone()),
        );

        self.operations
            .push(MultipleAttributesOperationWithIndex::ToValues {
                operand: operand.clone(),
            });

        operand
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleAttributesOperationWithIndex::IsString);
    implement_assertion_operation!(is_int, MultipleAttributesOperationWithIndex::IsInt);
    implement_assertion_operation!(is_max, MultipleAttributesOperationWithIndex::IsMax);
    implement_assertion_operation!(is_min, MultipleAttributesOperationWithIndex::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
    {
        let mut either_operand = Wrapper::<MultipleAttributesOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<MultipleAttributesOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleAttributesOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
    {
        let mut operand = Wrapper::<MultipleAttributesOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperationWithIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<MultipleAttributesOperandWithIndex<O>> {
    pub(crate) fn new(context: MultipleAttributesWithIndexContext<O>, kind: MultipleKind) -> Self {
        MultipleAttributesOperandWithIndex::new(context, kind).into()
    }

    implement_wrapper_operand_with_return!(max, SingleAttributeOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(min, SingleAttributeOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(count, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(sum, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(random, SingleAttributeOperandWithIndex<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleAttributesComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    implement_wrapper_operand_with_return!(to_values, MultipleValuesOperandWithIndex<O>);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperandWithIndex<O>>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

pub type NodeMultipleAttributesOperandWithoutIndex =
    MultipleAttributesOperandWithoutIndex<NodeOperand>;
pub type EdgeMultipleAttributesOperandWithoutIndex =
    MultipleAttributesOperandWithoutIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleAttributesOperandWithoutIndex<O: RootOperand> {
    context: MultipleAttributesWithoutIndexContext<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleAttributesOperandWithoutIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleAttributesOperandWithoutIndex<O> {
    type InputValue = BoxedIterator<'a, MedRecordAttribute>;
    type ReturnValue = BoxedIterator<'a, MedRecordAttribute>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        attributes: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let attributes = Box::new(attributes) as BoxedIterator<_>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a>
    for MultipleAttributesOperandWithoutIndex<O>
{
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate_grouped(medrecord, attribute_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleAttributesOperandWithoutIndex<O> {
    type ReturnValue = BoxedIterator<'a, MedRecordAttribute>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.get_attributes(medrecord, &self.kind)?;

        self.evaluate_forward(medrecord, attributes)
    }
}

impl<O: RootOperand> MultipleAttributesOperandWithoutIndex<O> {
    pub(crate) fn new(
        context: MultipleAttributesWithoutIndexContext<O>,
        kind: MultipleKind,
    ) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_attribute_operation_without_index!(max, Max, WithoutIndex);
    implement_attribute_operation_without_index!(min, Min, WithoutIndex);
    implement_attribute_operation_without_index!(count, Count, WithoutIndex);
    implement_attribute_operation_without_index!(sum, Sum, WithoutIndex);
    implement_attribute_operation_without_index!(random, Random, WithoutIndex);

    implement_single_attribute_comparison_operation!(
        greater_than,
        MultipleAttributesOperationWithoutIndex,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        MultipleAttributesOperationWithoutIndex,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        less_than,
        MultipleAttributesOperationWithoutIndex,
        LessThan
    );
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        MultipleAttributesOperationWithoutIndex,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        equal_to,
        MultipleAttributesOperationWithoutIndex,
        EqualTo
    );
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        MultipleAttributesOperationWithoutIndex,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        MultipleAttributesOperationWithoutIndex,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(
        ends_with,
        MultipleAttributesOperationWithoutIndex,
        EndsWith
    );
    implement_single_attribute_comparison_operation!(
        contains,
        MultipleAttributesOperationWithoutIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, MultipleAttributesOperationWithoutIndex, Add);
    implement_binary_arithmetic_operation!(sub, MultipleAttributesOperationWithoutIndex, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleAttributesOperationWithoutIndex, Mul);
    implement_binary_arithmetic_operation!(pow, MultipleAttributesOperationWithoutIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleAttributesOperationWithoutIndex, Mod);

    implement_unary_arithmetic_operation!(abs, MultipleAttributesOperationWithoutIndex, Abs);
    implement_unary_arithmetic_operation!(trim, MultipleAttributesOperationWithoutIndex, Trim);
    implement_unary_arithmetic_operation!(
        trim_start,
        MultipleAttributesOperationWithoutIndex,
        TrimStart
    );
    implement_unary_arithmetic_operation!(
        trim_end,
        MultipleAttributesOperationWithoutIndex,
        TrimEnd
    );
    implement_unary_arithmetic_operation!(
        lowercase,
        MultipleAttributesOperationWithoutIndex,
        Lowercase
    );
    implement_unary_arithmetic_operation!(
        uppercase,
        MultipleAttributesOperationWithoutIndex,
        Uppercase
    );

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleAttributesOperationWithoutIndex::IsString);
    implement_assertion_operation!(is_int, MultipleAttributesOperationWithoutIndex::IsInt);
    implement_assertion_operation!(is_max, MultipleAttributesOperationWithoutIndex::IsMax);
    implement_assertion_operation!(is_min, MultipleAttributesOperationWithoutIndex::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
    {
        let mut either_operand = Wrapper::<MultipleAttributesOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<MultipleAttributesOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleAttributesOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
    {
        let mut operand = Wrapper::<MultipleAttributesOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperationWithoutIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<MultipleAttributesOperandWithoutIndex<O>> {
    pub(crate) fn new(
        context: MultipleAttributesWithoutIndexContext<O>,
        kind: MultipleKind,
    ) -> Self {
        MultipleAttributesOperandWithoutIndex::new(context, kind).into()
    }

    implement_wrapper_operand_with_return!(max, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(min, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(count, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(sum, SingleAttributeOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(random, SingleAttributeOperandWithoutIndex<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleAttributesComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

pub type NodeSingleAttributeOperandWithIndex = SingleAttributeOperandWithIndex<NodeOperand>;
pub type EdgeSingleAttributeOperandWithIndex = SingleAttributeOperandWithIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleAttributeOperandWithIndex<O: RootOperand> {
    context: MultipleAttributesOperandWithIndex<O>,
    pub(crate) kind: SingleKindWithIndex,
    operations: Vec<SingleAttributeOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleAttributeOperandWithIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleAttributeOperandWithIndex<O> {
    type InputValue = Option<(&'a O::Index, MedRecordAttribute)>;
    type ReturnValue = Option<(&'a O::Index, MedRecordAttribute)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        attribute: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(attribute, |attribute, operation| {
                operation.evaluate(medrecord, attribute)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleAttributeOperandWithIndex<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(attributes, |attributes, operation| {
                operation.evaluate_grouped(medrecord, attributes)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleAttributeOperandWithIndex<O> {
    type ReturnValue = <Self as EvaluateForward<'a>>::ReturnValue;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.evaluate_backward(medrecord)?;

        let attribute = self.reduce_input(attributes)?;

        self.evaluate_forward(medrecord, attribute)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleAttributeOperandWithIndex<O> {
    type Context = MultipleAttributesOperandWithIndex<O>;

    #[inline]
    fn reduce_input(
        &self,
        attributes: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKindWithIndex::Max => {
                MultipleAttributesOperationWithIndex::<O>::get_max(attributes)?
            }
            SingleKindWithIndex::Min => {
                MultipleAttributesOperationWithIndex::<O>::get_min(attributes)?
            }
            SingleKindWithIndex::Random => {
                MultipleAttributesOperationWithIndex::<O>::get_random(attributes)
            }
        })
    }
}

impl<O: RootOperand> SingleAttributeOperandWithIndex<O> {
    pub(crate) fn new(
        context: MultipleAttributesOperandWithIndex<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_single_attribute_comparison_operation!(
        greater_than,
        SingleAttributeOperationWithIndex,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        SingleAttributeOperationWithIndex,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        less_than,
        SingleAttributeOperationWithIndex,
        LessThan
    );
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        SingleAttributeOperationWithIndex,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        equal_to,
        SingleAttributeOperationWithIndex,
        EqualTo
    );
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        SingleAttributeOperationWithIndex,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        SingleAttributeOperationWithIndex,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(
        ends_with,
        SingleAttributeOperationWithIndex,
        EndsWith
    );
    implement_single_attribute_comparison_operation!(
        contains,
        SingleAttributeOperationWithIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, SingleAttributeOperationWithIndex, Add);
    implement_binary_arithmetic_operation!(sub, SingleAttributeOperationWithIndex, Sub);
    implement_binary_arithmetic_operation!(mul, SingleAttributeOperationWithIndex, Mul);
    implement_binary_arithmetic_operation!(pow, SingleAttributeOperationWithIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleAttributeOperationWithIndex, Mod);

    implement_unary_arithmetic_operation!(abs, SingleAttributeOperationWithIndex, Abs);
    implement_unary_arithmetic_operation!(trim, SingleAttributeOperationWithIndex, Trim);
    implement_unary_arithmetic_operation!(trim_start, SingleAttributeOperationWithIndex, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, SingleAttributeOperationWithIndex, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, SingleAttributeOperationWithIndex, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, SingleAttributeOperationWithIndex, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleAttributeOperationWithIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleAttributeOperationWithIndex::IsString);
    implement_assertion_operation!(is_int, SingleAttributeOperationWithIndex::IsInt);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
    {
        let mut either_operand = Wrapper::<SingleAttributeOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<SingleAttributeOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleAttributeOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
    {
        let mut operand = Wrapper::<SingleAttributeOperandWithIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperationWithIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleAttributeOperandWithIndex<O>> {
    pub(crate) fn new(
        context: MultipleAttributesOperandWithIndex<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        SingleAttributeOperandWithIndex::new(context, kind).into()
    }

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleAttributesComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperandWithIndex<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

pub type NodeSingleAttributeOperandWithoutIndex = SingleAttributeOperandWithoutIndex<NodeOperand>;
pub type EdgeSingleAttributeOperandWithoutIndex = SingleAttributeOperandWithoutIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleAttributeOperandWithoutIndex<O: RootOperand> {
    context: SingleAttributeWithoutIndexContext<O>,
    pub(crate) kind: SingleKindWithoutIndex,
    operations: Vec<SingleAttributeOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleAttributeOperandWithoutIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleAttributeOperandWithoutIndex<O> {
    type InputValue = Option<MedRecordAttribute>;
    type ReturnValue = Option<MedRecordAttribute>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        attribute: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(attribute, |attribute, operation| {
                operation.evaluate(medrecord, attribute)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleAttributeOperandWithoutIndex<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(attributes, |attributes, operation| {
                operation.evaluate_grouped(medrecord, attributes)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleAttributeOperandWithoutIndex<O> {
    type ReturnValue = <Self as EvaluateForward<'a>>::ReturnValue;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.get_attributes(medrecord)?;

        let attribute = match self.kind {
            SingleKindWithoutIndex::Max => {
                MultipleAttributesOperationWithoutIndex::<O>::get_max(attributes)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleAttributesOperationWithoutIndex::<O>::get_min(attributes)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleAttributesOperationWithoutIndex::<O>::get_count(attributes),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleAttributesOperationWithoutIndex::<O>::get_sum(attributes)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleAttributesOperationWithoutIndex::<O>::get_random(attributes)
            }
        };

        self.evaluate_forward(medrecord, attribute)
    }
}

impl<O: RootOperand> SingleAttributeOperandWithoutIndex<O> {
    pub(crate) fn new(
        context: SingleAttributeWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_single_attribute_comparison_operation!(
        greater_than,
        SingleAttributeOperationWithoutIndex,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        SingleAttributeOperationWithoutIndex,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        less_than,
        SingleAttributeOperationWithoutIndex,
        LessThan
    );
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        SingleAttributeOperationWithoutIndex,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        equal_to,
        SingleAttributeOperationWithoutIndex,
        EqualTo
    );
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        SingleAttributeOperationWithoutIndex,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        SingleAttributeOperationWithoutIndex,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(
        ends_with,
        SingleAttributeOperationWithoutIndex,
        EndsWith
    );
    implement_single_attribute_comparison_operation!(
        contains,
        SingleAttributeOperationWithoutIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, SingleAttributeOperationWithoutIndex, Add);
    implement_binary_arithmetic_operation!(sub, SingleAttributeOperationWithoutIndex, Sub);
    implement_binary_arithmetic_operation!(mul, SingleAttributeOperationWithoutIndex, Mul);
    implement_binary_arithmetic_operation!(pow, SingleAttributeOperationWithoutIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleAttributeOperationWithoutIndex, Mod);

    implement_unary_arithmetic_operation!(abs, SingleAttributeOperationWithoutIndex, Abs);
    implement_unary_arithmetic_operation!(trim, SingleAttributeOperationWithoutIndex, Trim);
    implement_unary_arithmetic_operation!(
        trim_start,
        SingleAttributeOperationWithoutIndex,
        TrimStart
    );
    implement_unary_arithmetic_operation!(trim_end, SingleAttributeOperationWithoutIndex, TrimEnd);
    implement_unary_arithmetic_operation!(
        lowercase,
        SingleAttributeOperationWithoutIndex,
        Lowercase
    );
    implement_unary_arithmetic_operation!(
        uppercase,
        SingleAttributeOperationWithoutIndex,
        Uppercase
    );

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleAttributeOperationWithoutIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleAttributeOperationWithoutIndex::IsString);
    implement_assertion_operation!(is_int, SingleAttributeOperationWithoutIndex::IsInt);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
    {
        let mut either_operand = Wrapper::<SingleAttributeOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<SingleAttributeOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleAttributeOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
    {
        let mut operand = Wrapper::<SingleAttributeOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperationWithoutIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleAttributeOperandWithoutIndex<O>> {
    pub(crate) fn new(
        context: SingleAttributeWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
        SingleAttributeOperandWithoutIndex::new(context, kind).into()
    }

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleAttributesComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

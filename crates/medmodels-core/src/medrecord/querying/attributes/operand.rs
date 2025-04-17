use medmodels_utils::aliases::MrHashSet;

use super::{
    operation::{AttributesTreeOperation, MultipleAttributesOperation, SingleAttributeOperation},
    BinaryArithmeticKind, MultipleComparisonKind, MultipleKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            BoxedIterator, Index, Operand, OptionalIndexWrapper,
        },
        EdgeOperand, MedRecordAttribute, NodeOperand, Wrapper,
    },
    MedRecord,
};
use std::collections::HashSet;

macro_rules! implement_attributes_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<MultipleAttributesOperand<O>> {
            let operand = Wrapper::<MultipleAttributesOperand<O>>::new(
                self.deep_clone(),
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

macro_rules! implement_attribute_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleAttributeOperand<O>> {
            let operand =
                Wrapper::<SingleAttributeOperand<O>>::new(self.deep_clone(), SingleKind::$variant);

            self.operations
                .push(MultipleAttributesOperation::AttributeOperation {
                    operand: operand.clone(),
                });

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
    NodeSingleAttributeOperand(NodeSingleAttributeOperand),
    EdgeSingleAttributeOperand(EdgeSingleAttributeOperand),
    Attribute(MedRecordAttribute),
}

impl DeepClone for SingleAttributeComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeSingleAttributeOperand(operand) => {
                Self::NodeSingleAttributeOperand(operand.deep_clone())
            }
            Self::EdgeSingleAttributeOperand(operand) => {
                Self::EdgeSingleAttributeOperand(operand.deep_clone())
            }
            Self::Attribute(attribute) => Self::Attribute(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeSingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<NodeSingleAttributeOperand>) -> Self {
        Self::NodeSingleAttributeOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<NodeSingleAttributeOperand>) -> Self {
        Self::NodeSingleAttributeOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<EdgeSingleAttributeOperand>) -> Self {
        Self::EdgeSingleAttributeOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleAttributeOperand>) -> Self {
        Self::EdgeSingleAttributeOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeSingleAttributeOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.unpack().1),
            Self::EdgeSingleAttributeOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.unpack().1),
            Self::Attribute(attribute) => Some(attribute.clone()),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesComparisonOperand {
    NodeMultipleAttributesOperand(NodeMultipleAttributesOperand),
    EdgeMultipleAttributesOperand(EdgeMultipleAttributesOperand),
    Attributes(MrHashSet<MedRecordAttribute>),
}

impl DeepClone for MultipleAttributesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeMultipleAttributesOperand(operand) => {
                Self::NodeMultipleAttributesOperand(operand.deep_clone())
            }
            Self::EdgeMultipleAttributesOperand(operand) => {
                Self::EdgeMultipleAttributesOperand(operand.deep_clone())
            }
            Self::Attributes(attribute) => Self::Attributes(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeMultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleAttributesOperand>) -> Self {
        Self::NodeMultipleAttributesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: &Wrapper<NodeMultipleAttributesOperand>) -> Self {
        Self::NodeMultipleAttributesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleAttributesOperand>) -> Self {
        Self::EdgeMultipleAttributesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: &Wrapper<EdgeMultipleAttributesOperand>) -> Self {
        Self::EdgeMultipleAttributesOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeMultipleAttributesOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::EdgeMultipleAttributesOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::Attributes(attributes) => attributes.clone(),
        })
    }
}

pub type NodeAttributesTreeOperand = AttributesTreeOperand<NodeOperand>;
pub type EdgeAttributesTreeOperand = AttributesTreeOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct AttributesTreeOperand<O: Operand> {
    pub(crate) context: O,
    operations: Vec<AttributesTreeOperation<O>>,
}

impl<O: Operand> DeepClone for AttributesTreeOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<O: Operand> AttributesTreeOperand<O> {
    pub(crate) fn new(context: O) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward<'a, I: Index + 'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (I, Vec<MedRecordAttribute>)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (I, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        let attributes = Box::new(attributes) as BoxedIterator<(I, Vec<MedRecordAttribute>)>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        let attributes: BoxedIterator<_> = Box::new(self.context.get_attributes(medrecord)?);

        self.evaluate_forward(medrecord, attributes)
    }

    implement_attributes_operation!(max, Max);
    implement_attributes_operation!(min, Min);
    implement_attributes_operation!(count, Count);
    implement_attributes_operation!(sum, Sum);
    implement_attributes_operation!(first, First);
    implement_attributes_operation!(last, Last);

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

impl<O: Operand> Wrapper<AttributesTreeOperand<O>> {
    pub(crate) fn new(context: O) -> Self {
        AttributesTreeOperand::new(context).into()
    }

    pub(crate) fn evaluate_forward<'a, I: Index + 'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (I, Vec<MedRecordAttribute>)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (I, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        self.0
            .read_or_panic()
            .evaluate_forward(medrecord, attributes)
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        self.0.read_or_panic().evaluate_backward(medrecord)
    }

    implement_wrapper_operand_with_return!(max, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(min, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(count, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(sum, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(first, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(last, MultipleAttributesOperand<O>);

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

pub type NodeMultipleAttributesOperand = MultipleAttributesOperand<NodeOperand>;
pub type EdgeMultipleAttributesOperand = MultipleAttributesOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleAttributesOperand<O: Operand> {
    pub(crate) context: AttributesTreeOperand<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperation<O>>,
}

impl<O: Operand> DeepClone for MultipleAttributesOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<O: Operand> MultipleAttributesOperand<O> {
    pub(crate) fn new(context: AttributesTreeOperand<O>, kind: MultipleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward<'a, I: Index + 'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (I, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (I, MedRecordAttribute)> + 'a>
    where
        O: 'a,
    {
        let attributes = Box::new(attributes) as BoxedIterator<(I, MedRecordAttribute)>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a>
    where
        O: 'a,
    {
        let attributes = self.context.evaluate_backward(medrecord)?;

        let attributes: BoxedIterator<_> = match self.kind {
            MultipleKind::Max => Box::new(AttributesTreeOperation::<O>::get_max(attributes)?),
            MultipleKind::Min => Box::new(AttributesTreeOperation::<O>::get_min(attributes)?),
            MultipleKind::Count => Box::new(AttributesTreeOperation::<O>::get_count(attributes)?),
            MultipleKind::Sum => Box::new(AttributesTreeOperation::<O>::get_sum(attributes)?),
            MultipleKind::First => Box::new(AttributesTreeOperation::<O>::get_first(attributes)?),
            MultipleKind::Last => Box::new(AttributesTreeOperation::<O>::get_last(attributes)?),
        };

        self.evaluate_forward(medrecord, attributes)
    }

    implement_attribute_operation!(max, Max);
    implement_attribute_operation!(min, Min);
    implement_attribute_operation!(count, Count);
    implement_attribute_operation!(sum, Sum);
    implement_attribute_operation!(first, First);
    implement_attribute_operation!(last, Last);

    implement_single_attribute_comparison_operation!(
        greater_than,
        MultipleAttributesOperation,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        MultipleAttributesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        less_than,
        MultipleAttributesOperation,
        LessThan
    );
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        MultipleAttributesOperation,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(
        equal_to,
        MultipleAttributesOperation,
        EqualTo
    );
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        MultipleAttributesOperation,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        MultipleAttributesOperation,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(
        ends_with,
        MultipleAttributesOperation,
        EndsWith
    );
    implement_single_attribute_comparison_operation!(
        contains,
        MultipleAttributesOperation,
        Contains
    );

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, MultipleAttributesOperation, Add);
    implement_binary_arithmetic_operation!(sub, MultipleAttributesOperation, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleAttributesOperation, Mul);
    implement_binary_arithmetic_operation!(pow, MultipleAttributesOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleAttributesOperation, Mod);

    implement_unary_arithmetic_operation!(abs, MultipleAttributesOperation, Abs);
    implement_unary_arithmetic_operation!(trim, MultipleAttributesOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, MultipleAttributesOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, MultipleAttributesOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, MultipleAttributesOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, MultipleAttributesOperation, Uppercase);

    #[allow(clippy::wrong_self_convention)]
    pub fn to_values(&mut self) -> Wrapper<MultipleValuesOperand<O>> {
        let operand = Wrapper::<MultipleValuesOperand<O>>::new(
            values::Context::MultipleAttributesOperand(self.deep_clone()),
        );

        self.operations.push(MultipleAttributesOperation::ToValues {
            operand: operand.clone(),
        });

        operand
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleAttributesOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleAttributesOperation::IsString);
    implement_assertion_operation!(is_int, MultipleAttributesOperation::IsInt);
    implement_assertion_operation!(is_max, MultipleAttributesOperation::IsMax);
    implement_assertion_operation!(is_min, MultipleAttributesOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<MultipleAttributesOperand<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<MultipleAttributesOperand<O>>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(MultipleAttributesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
    {
        let mut operand =
            Wrapper::<MultipleAttributesOperand<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperation::Exclude { operand });
    }
}

impl<O: Operand> Wrapper<MultipleAttributesOperand<O>> {
    pub(crate) fn new(context: AttributesTreeOperand<O>, kind: MultipleKind) -> Self {
        MultipleAttributesOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate_forward<'a, I: Index + 'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (I, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (I, MedRecordAttribute)> + 'a>
    where
        O: 'a,
    {
        self.0
            .read_or_panic()
            .evaluate_forward(medrecord, attributes)
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a>
    where
        O: 'a,
    {
        self.0.read_or_panic().evaluate_backward(medrecord)
    }

    implement_wrapper_operand_with_return!(max, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(min, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(count, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(first, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(last, SingleAttributeOperand<O>);

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

    implement_wrapper_operand_with_return!(to_values, MultipleValuesOperand<O>);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperand<O>>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

pub type NodeSingleAttributeOperand = SingleAttributeOperand<NodeOperand>;
pub type EdgeSingleAttributeOperand = SingleAttributeOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleAttributeOperand<O: Operand> {
    pub(crate) context: MultipleAttributesOperand<O>,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleAttributeOperation<O>>,
}

impl<O: Operand> DeepClone for SingleAttributeOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<O: Operand> SingleAttributeOperand<O> {
    pub(crate) fn new(context: MultipleAttributesOperand<O>, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward<I: Index>(
        &self,
        medrecord: &MedRecord,
        attribute: OptionalIndexWrapper<I, MedRecordAttribute>,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordAttribute>>> {
        self.operations
            .iter()
            .try_fold(Some(attribute), |attribute, operation| {
                if let Some(attribute) = attribute {
                    operation.evaluate(medrecord, attribute)
                } else {
                    Ok(None)
                }
            })
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<&'a O::Index, MedRecordAttribute>>>
    where
        O: 'a,
    {
        let attributes = self.context.evaluate_backward(medrecord)?;

        let attribute: OptionalIndexWrapper<_, _> = match self.kind {
            SingleKind::Max => MultipleAttributesOperation::<O>::get_max(attributes)?.into(),
            SingleKind::Min => MultipleAttributesOperation::<O>::get_min(attributes)?.into(),
            SingleKind::Count => MultipleAttributesOperation::<O>::get_count(attributes).into(),
            SingleKind::Sum => MultipleAttributesOperation::<O>::get_sum(attributes)?.into(),
            SingleKind::First => MultipleAttributesOperation::<O>::get_first(attributes)?.into(),
            SingleKind::Last => MultipleAttributesOperation::<O>::get_last(attributes)?.into(),
        };

        self.evaluate_forward(medrecord, attribute)
    }

    implement_single_attribute_comparison_operation!(
        greater_than,
        SingleAttributeOperation,
        GreaterThan
    );
    implement_single_attribute_comparison_operation!(
        greater_than_or_equal_to,
        SingleAttributeOperation,
        GreaterThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(less_than, SingleAttributeOperation, LessThan);
    implement_single_attribute_comparison_operation!(
        less_than_or_equal_to,
        SingleAttributeOperation,
        LessThanOrEqualTo
    );
    implement_single_attribute_comparison_operation!(equal_to, SingleAttributeOperation, EqualTo);
    implement_single_attribute_comparison_operation!(
        not_equal_to,
        SingleAttributeOperation,
        NotEqualTo
    );
    implement_single_attribute_comparison_operation!(
        starts_with,
        SingleAttributeOperation,
        StartsWith
    );
    implement_single_attribute_comparison_operation!(ends_with, SingleAttributeOperation, EndsWith);
    implement_single_attribute_comparison_operation!(contains, SingleAttributeOperation, Contains);

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, SingleAttributeOperation, Add);
    implement_binary_arithmetic_operation!(sub, SingleAttributeOperation, Sub);
    implement_binary_arithmetic_operation!(mul, SingleAttributeOperation, Mul);
    implement_binary_arithmetic_operation!(pow, SingleAttributeOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleAttributeOperation, Mod);

    implement_unary_arithmetic_operation!(abs, SingleAttributeOperation, Abs);
    implement_unary_arithmetic_operation!(trim, SingleAttributeOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, SingleAttributeOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, SingleAttributeOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, SingleAttributeOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, SingleAttributeOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleAttributeOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleAttributeOperation::IsString);
    implement_assertion_operation!(is_int, SingleAttributeOperation::IsInt);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<SingleAttributeOperand<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleAttributeOperand<O>>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(SingleAttributeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
    {
        let mut operand =
            Wrapper::<SingleAttributeOperand<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperation::Exclude { operand });
    }
}

impl<O: Operand> Wrapper<SingleAttributeOperand<O>> {
    pub(crate) fn new(context: MultipleAttributesOperand<O>, kind: SingleKind) -> Self {
        SingleAttributeOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate_forward<I: Index>(
        &self,
        medrecord: &MedRecord,
        attribute: OptionalIndexWrapper<I, MedRecordAttribute>,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordAttribute>>> {
        self.0
            .read_or_panic()
            .evaluate_forward(medrecord, attribute)
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<&'a O::Index, MedRecordAttribute>>>
    where
        O: 'a,
    {
        self.0.read_or_panic().evaluate_backward(medrecord)
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
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

use super::{
    operation::{AttributesTreeOperation, MultipleAttributesOperation, SingleAttributeOperation},
    BinaryArithmeticKind, Context, GetAttributes, MultipleComparisonKind, MultipleKind,
    SingleComparisonKind, SingleKind, UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            BoxedIterator,
        },
        MedRecordAttribute, Wrapper,
    },
    MedRecord,
};
use std::{fmt::Display, hash::Hash};

macro_rules! implement_attributes_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<MultipleAttributesOperand> {
            let operand = Wrapper::<MultipleAttributesOperand>::new(
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
        pub fn $name(&mut self) -> Wrapper<SingleAttributeOperand> {
            let operand =
                Wrapper::<SingleAttributeOperand>::new(self.deep_clone(), SingleKind::$variant);

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
    ($name:ident, $return_operand:ident) => {
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
    Operand(SingleAttributeOperand),
    Attribute(MedRecordAttribute),
}

impl DeepClone for SingleAttributeComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Attribute(attribute) => Self::Attribute(attribute.clone()),
        }
    }
}

impl From<Wrapper<SingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<SingleAttributeOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<SingleAttributeOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<SingleAttributeOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>> From<V> for SingleAttributeComparisonOperand {
    fn from(value: V) -> Self {
        Self::Attribute(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesComparisonOperand {
    Operand(MultipleAttributesOperand),
    Attributes(Vec<MedRecordAttribute>),
}

impl DeepClone for MultipleAttributesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Attributes(attribute) => Self::Attributes(attribute.clone()),
        }
    }
}

impl From<Wrapper<MultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<MultipleAttributesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<MultipleAttributesOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: &Wrapper<MultipleAttributesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>> From<Vec<V>> for MultipleAttributesComparisonOperand {
    fn from(value: Vec<V>) -> Self {
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

#[derive(Debug, Clone)]
pub struct AttributesTreeOperand {
    pub(crate) context: Context,
    operations: Vec<AttributesTreeOperation>,
}

impl DeepClone for AttributesTreeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl AttributesTreeOperand {
    pub(crate) fn new(context: Context) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (, Vec<MedRecordAttribute>)> + 'a> {
        let attributes = self.context.get_attributes(medrecord)?;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
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
        EQ: FnOnce(&mut Wrapper<AttributesTreeOperand>),
        OQ: FnOnce(&mut Wrapper<AttributesTreeOperand>),
    {
        let mut either_operand = Wrapper::<AttributesTreeOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<AttributesTreeOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(AttributesTreeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<AttributesTreeOperand>),
    {
        let mut operand = Wrapper::<AttributesTreeOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(AttributesTreeOperation::Exclude { operand });
    }
}

impl Wrapper<AttributesTreeOperand> {
    pub(crate) fn new(context: Context) -> Self {
        AttributesTreeOperand::new(context).into()
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (T, Vec<MedRecordAttribute>)> + 'a> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_return!(max, MultipleAttributesOperand);
    implement_wrapper_operand_with_return!(min, MultipleAttributesOperand);
    implement_wrapper_operand_with_return!(count, MultipleAttributesOperand);
    implement_wrapper_operand_with_return!(sum, MultipleAttributesOperand);
    implement_wrapper_operand_with_return!(first, MultipleAttributesOperand);
    implement_wrapper_operand_with_return!(last, MultipleAttributesOperand);

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
        EQ: FnOnce(&mut Wrapper<AttributesTreeOperand>),
        OQ: FnOnce(&mut Wrapper<AttributesTreeOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<AttributesTreeOperand>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

#[derive(Debug, Clone)]
pub struct MultipleAttributesOperand {
    pub(crate) context: AttributesTreeOperand,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperation>,
}

impl DeepClone for MultipleAttributesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl MultipleAttributesOperand {
    pub(crate) fn new(context: AttributesTreeOperand, kind: MultipleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)> + 'a> {
        let attributes = Box::new(attributes) as BoxedIterator<(T, MedRecordAttribute)>;

        self.operations
            .iter()
            .try_fold(attributes, |attribute_tuples, operation| {
                operation.evaluate(medrecord, attribute_tuples)
            })
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
    pub fn to_values(&mut self) -> Wrapper<MultipleValuesOperand> {
        let operand = Wrapper::<MultipleValuesOperand>::new(
            values::Context::MultipleAttributesOperand(self.deep_clone()),
            "unused".into(),
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
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
    {
        let mut either_operand =
            Wrapper::<MultipleAttributesOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<MultipleAttributesOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(MultipleAttributesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
    {
        let mut operand =
            Wrapper::<MultipleAttributesOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperation::Exclude { operand });
    }
}

impl Wrapper<MultipleAttributesOperand> {
    pub(crate) fn new(context: AttributesTreeOperand, kind: MultipleKind) -> Self {
        MultipleAttributesOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)> + 'a> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_return!(max, SingleAttributeOperand);
    implement_wrapper_operand_with_return!(min, SingleAttributeOperand);
    implement_wrapper_operand_with_return!(count, SingleAttributeOperand);
    implement_wrapper_operand_with_return!(sum, SingleAttributeOperand);
    implement_wrapper_operand_with_return!(first, SingleAttributeOperand);
    implement_wrapper_operand_with_return!(last, SingleAttributeOperand);

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

    implement_wrapper_operand_with_return!(to_values, MultipleValuesOperand);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
        OQ: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleAttributesOperand>),
    {
        self.0.write_or_panic().exclude(query)
    }
}

#[derive(Debug, Clone)]
pub struct SingleAttributeOperand {
    pub(crate) context: MultipleAttributesOperand,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleAttributeOperation>,
}

impl DeepClone for SingleAttributeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl SingleAttributeOperand {
    pub(crate) fn new(context: MultipleAttributesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
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
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperand>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperand>),
    {
        let mut either_operand =
            Wrapper::<SingleAttributeOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleAttributeOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(SingleAttributeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperand>),
    {
        let mut operand =
            Wrapper::<SingleAttributeOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperation::Exclude { operand });
    }
}

impl Wrapper<SingleAttributeOperand> {
    pub(crate) fn new(context: MultipleAttributesOperand, kind: SingleKind) -> Self {
        SingleAttributeOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        self.0.read_or_panic().evaluate(medrecord)
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
        EQ: FnOnce(&mut Wrapper<SingleAttributeOperand>),
        OQ: FnOnce(&mut Wrapper<SingleAttributeOperand>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleAttributeOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

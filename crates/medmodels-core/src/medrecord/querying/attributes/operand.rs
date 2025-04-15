use super::{
    operation::{AttributesTreeOperation, MultipleAttributesOperation, SingleAttributeOperation},
    BinaryArithmeticKind, MultipleComparisonKind, MultipleKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            BoxedIterator, Operand,
        },
        DataType, MedRecordAttribute, Wrapper,
    },
    MedRecord,
};
use std::{cmp::Ordering, ops::Add};

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
        pub fn $name<V: Into<SingleAttributeComparisonOperand<O>>>(&mut self, attribute: V) {
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
        pub fn $name<V: Into<SingleAttributeComparisonOperand<O>>>(&mut self, attribute: V) {
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
pub enum SingleAttributeComparisonOperand<O: Operand> {
    Operand(SingleAttributeOperand<O>),
    Attribute(MedRecordAttribute),
}

impl<O: Operand> DeepClone for SingleAttributeComparisonOperand<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Attribute(attribute) => Self::Attribute(attribute.clone()),
        }
    }
}

impl<O: Operand> From<Wrapper<SingleAttributeOperand<O>>> for SingleAttributeComparisonOperand<O> {
    fn from(value: Wrapper<SingleAttributeOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<O: Operand> From<&Wrapper<SingleAttributeOperand<O>>> for SingleAttributeComparisonOperand<O> {
    fn from(value: &Wrapper<SingleAttributeOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>, O: Operand> From<V> for SingleAttributeComparisonOperand<O> {
    fn from(value: V) -> Self {
        Self::Attribute(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesComparisonOperand<O: Operand> {
    Operand(MultipleAttributesOperand<O>),
    Attributes(Vec<MedRecordAttribute>),
}

impl<O: Operand> DeepClone for MultipleAttributesComparisonOperand<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Attributes(attribute) => Self::Attributes(attribute.clone()),
        }
    }
}

impl<O: Operand> From<Wrapper<MultipleAttributesOperand<O>>>
    for MultipleAttributesComparisonOperand<O>
{
    fn from(value: Wrapper<MultipleAttributesOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<O: Operand> From<&Wrapper<MultipleAttributesOperand<O>>>
    for MultipleAttributesComparisonOperand<O>
{
    fn from(value: &Wrapper<MultipleAttributesOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordAttribute>, O: Operand> From<Vec<V>>
    for MultipleAttributesComparisonOperand<O>
{
    fn from(value: Vec<V>) -> Self {
        Self::Attributes(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordAttribute> + Clone, O: Operand, const N: usize> From<[V; N]>
    for MultipleAttributesComparisonOperand<O>
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

#[derive(Debug, Clone)]
pub struct AttributesTreeOperand<O: Operand> {
    pub(crate) context: O,
    operations: Vec<AttributesTreeOperation>,
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>> {
        let attributes = Box::new(self.context.get_attributes(medrecord)?)
            as BoxedIterator<(&'a O::Index, Vec<MedRecordAttribute>)>;

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

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
        self.operations.push(
            AttributesTreeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_return!(max, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(min, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(count, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(sum, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(first, MultipleAttributesOperand<O>);
    implement_wrapper_operand_with_return!(last, MultipleAttributesOperand<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        contains,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand<O>>);

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

#[derive(Debug, Clone)]
pub struct MultipleAttributesOperand<O: Operand> {
    pub(crate) context: AttributesTreeOperand<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperation>,
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

    #[inline]
    pub(crate) fn get_max<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |max, attribute| {
                match attribute.partial_cmp(&max) {
                    Some(Ordering::Greater) => Ok(attribute),
                    None => {
                        let first_dtype = DataType::from(attribute);
                        let second_dtype = DataType::from(max);

                        Err(MedRecordError::QueryError(format!(
                            "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                            first_dtype, second_dtype
                        )))
                    }
                    _ => Ok(max),
                }
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_min<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |max, attribute| {
                match attribute.partial_cmp(&max) {
                    Some(Ordering::Less) => Ok(attribute),
                    None => {
                        let first_dtype = DataType::from(attribute);
                        let second_dtype = DataType::from(max);

                        Err(MedRecordError::QueryError(format!(
                            "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                            first_dtype, second_dtype
                        )))
                    }
                    _ => Ok(max),
                }
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_count<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attribute)| (index, MedRecordAttribute::Int(attribute.len() as i64))))
    }

    #[inline]
    pub(crate) fn get_sum<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |sum, attribute| {
                let first_dtype = DataType::from(&sum);
                let second_dtype = DataType::from(&attribute);

                sum.add(attribute).map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Cannot add attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
                    ))
                })
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_first<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attributes)| {
                let first_attribute =
                    attributes
                        .into_iter()
                        .next()
                        .ok_or(MedRecordError::QueryError(
                            "No attributes to compare".to_string(),
                        ))?;

                Ok((index, first_attribute))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    pub(crate) fn get_last<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attributes)| {
                let first_attribute =
                    attributes
                        .into_iter()
                        .next_back()
                        .ok_or(MedRecordError::QueryError(
                            "No attributes to compare".to_string(),
                        ))?;

                Ok((index, first_attribute))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>> {
        let attributes = self.context.evaluate(medrecord)?;

        let attributes: BoxedIterator<(&O::Index, MedRecordAttribute)> = match self.kind {
            MultipleKind::Max => Box::new(Self::get_max(attributes)?),
            MultipleKind::Min => Box::new(Self::get_min(attributes)?),
            MultipleKind::Count => Box::new(Self::get_count(attributes)?),
            MultipleKind::Sum => Box::new(Self::get_sum(attributes)?),
            MultipleKind::First => Box::new(Self::get_first(attributes)?),
            MultipleKind::Last => Box::new(Self::get_last(attributes)?),
        };

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

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
        self.operations.push(
            MultipleAttributesOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_return!(max, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(min, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(count, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(first, SingleAttributeOperand<O>);
    implement_wrapper_operand_with_return!(last, SingleAttributeOperand<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        contains,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand<O>>);

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

#[derive(Debug, Clone)]
pub struct SingleAttributeOperand<O: Operand> {
    pub(crate) context: MultipleAttributesOperand<O>,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleAttributeOperation>,
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

    #[inline]
    pub(crate) fn get_max<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        let max_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(max_attribute, |max_attribute, attribute| {
            match attribute.1.partial_cmp(&max_attribute.1) {
                Some(Ordering::Greater) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute.1);
                    let second_dtype = DataType::from(max_attribute.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(max_attribute),
            }
        })
    }

    #[inline]
    pub(crate) fn get_min<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        let min_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(min_attribute, |min_attribute, attribute| {
            match attribute.1.partial_cmp(&min_attribute.1) {
                Some(Ordering::Less) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute.1);
                    let second_dtype = DataType::from(min_attribute.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(min_attribute),
            }
        })
    }

    #[inline]
    pub(crate) fn get_count<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordAttribute {
        MedRecordAttribute::Int(attributes.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<MedRecordAttribute> {
        let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(first_attribute.1, |sum, (_, attribute)| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&attribute);

            sum.add(attribute).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                    first_dtype, second_dtype
                ))
            })
        })
    }

    #[inline]
    pub(crate) fn get_first<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to get the first".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_last<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        attributes.last().ok_or(MedRecordError::QueryError(
            "No attributes to get the first".to_string(),
        ))
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let attributes = self.context.evaluate(medrecord)?;

        let attribute = match self.kind {
            SingleKind::Max => Self::get_max(attributes)?.1,
            SingleKind::Min => Self::get_min(attributes)?.1,
            SingleKind::Count => Self::get_count(attributes),
            SingleKind::Sum => Self::get_sum(attributes)?,
            SingleKind::First => Self::get_first(attributes)?.1,
            SingleKind::Last => Self::get_last(attributes)?.1,
        };

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

    pub fn is_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleAttributesComparisonOperand<O>>>(&mut self, attributes: V) {
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        ends_with,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        contains,
        impl Into<SingleAttributeComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleAttributesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleAttributeComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleAttributeComparisonOperand<O>>);

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

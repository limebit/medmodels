use super::{
    operation::{MultipleValuesOperation, SingleValueOperation},
    BinaryArithmeticKind, Context, MultipleComparisonKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            BoxedIterator, Operand, OptionalIndexWrapper,
        },
        MedRecordValue, Wrapper,
    },
    MedRecord,
};

macro_rules! implement_value_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueOperand<O>> {
            let operand =
                Wrapper::<SingleValueOperand<O>>::new(self.deep_clone(), SingleKind::$variant);

            self.operations
                .push(MultipleValuesOperation::ValueOperation {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_single_value_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<SingleValueComparisonOperand<O>>>(&mut self, value: V) {
            self.operations
                .push($operation::SingleValueComparisonOperation {
                    operand: value.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<SingleValueComparisonOperand<O>>>(&mut self, value: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: value.into(),
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
    ($name:ident, $value_type:ty) => {
        pub fn $name(&self, value: $value_type) {
            self.0.write_or_panic().$name(value)
        }
    };
}

#[derive(Debug, Clone)]
pub enum SingleValueComparisonOperand<O: Operand> {
    Operand(SingleValueOperand<O>),
    Value(MedRecordValue),
}

impl<O: Operand> DeepClone for SingleValueComparisonOperand<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Value(value) => Self::Value(value.clone()),
        }
    }
}

impl<O: Operand> From<Wrapper<SingleValueOperand<O>>> for SingleValueComparisonOperand<O> {
    fn from(value: Wrapper<SingleValueOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<O: Operand> From<&Wrapper<SingleValueOperand<O>>> for SingleValueComparisonOperand<O> {
    fn from(value: &Wrapper<SingleValueOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>, O: Operand> From<V> for SingleValueComparisonOperand<O> {
    fn from(value: V) -> Self {
        Self::Value(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesComparisonOperand<O: Operand> {
    Operand(MultipleValuesOperand<O>),
    Values(Vec<MedRecordValue>),
}

impl<O: Operand> DeepClone for MultipleValuesComparisonOperand<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Values(value) => Self::Values(value.clone()),
        }
    }
}

impl<O: Operand> From<Wrapper<MultipleValuesOperand<O>>> for MultipleValuesComparisonOperand<O> {
    fn from(value: Wrapper<MultipleValuesOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<O: Operand> From<&Wrapper<MultipleValuesOperand<O>>> for MultipleValuesComparisonOperand<O> {
    fn from(value: &Wrapper<MultipleValuesOperand<O>>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>, O: Operand> From<Vec<V>> for MultipleValuesComparisonOperand<O> {
    fn from(value: Vec<V>) -> Self {
        Self::Values(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordValue> + Clone, O: Operand, const N: usize> From<[V; N]>
    for MultipleValuesComparisonOperand<O>
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

#[derive(Debug, Clone)]
pub struct MultipleValuesOperand<O: Operand> {
    pub(crate) context: Context<O>,
    operations: Vec<MultipleValuesOperation<O>>,
}

impl<O: Operand> DeepClone for MultipleValuesOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<O: Operand> MultipleValuesOperand<O> {
    pub(crate) fn new(context: Context<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (O::Index, MedRecordValue)> + 'a>
    where
        O: 'a,
    {
        let values: BoxedIterator<(O::Index, MedRecordValue)> =
            Box::new(self.context.get_values(medrecord)?);

        self.operations
            .iter()
            .try_fold(values, |value_tuples, operation| {
                operation.evaluate(medrecord, value_tuples)
            })
    }

    implement_value_operation!(max, Max);
    implement_value_operation!(min, Min);
    implement_value_operation!(mean, Mean);
    implement_value_operation!(median, Median);
    implement_value_operation!(mode, Mode);
    implement_value_operation!(std, Std);
    implement_value_operation!(var, Var);
    implement_value_operation!(count, Count);
    implement_value_operation!(sum, Sum);
    implement_value_operation!(first, First);
    implement_value_operation!(last, Last);

    implement_single_value_comparison_operation!(
        greater_than,
        MultipleValuesOperation,
        GreaterThan
    );
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        MultipleValuesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, MultipleValuesOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        MultipleValuesOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, MultipleValuesOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, MultipleValuesOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, MultipleValuesOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, MultipleValuesOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, MultipleValuesOperation, Contains);

    pub fn is_in<V: Into<MultipleValuesComparisonOperand<O>>>(&mut self, values: V) {
        self.operations
            .push(MultipleValuesOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand<O>>>(&mut self, values: V) {
        self.operations
            .push(MultipleValuesOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, MultipleValuesOperation, Add);
    implement_binary_arithmetic_operation!(sub, MultipleValuesOperation, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleValuesOperation, Mul);
    implement_binary_arithmetic_operation!(div, MultipleValuesOperation, Div);
    implement_binary_arithmetic_operation!(pow, MultipleValuesOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleValuesOperation, Mod);

    implement_unary_arithmetic_operation!(round, MultipleValuesOperation, Round);
    implement_unary_arithmetic_operation!(ceil, MultipleValuesOperation, Ceil);
    implement_unary_arithmetic_operation!(floor, MultipleValuesOperation, Floor);
    implement_unary_arithmetic_operation!(abs, MultipleValuesOperation, Abs);
    implement_unary_arithmetic_operation!(sqrt, MultipleValuesOperation, Sqrt);
    implement_unary_arithmetic_operation!(trim, MultipleValuesOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, MultipleValuesOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, MultipleValuesOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, MultipleValuesOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, MultipleValuesOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleValuesOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleValuesOperation::IsString);
    implement_assertion_operation!(is_int, MultipleValuesOperation::IsInt);
    implement_assertion_operation!(is_float, MultipleValuesOperation::IsFloat);
    implement_assertion_operation!(is_bool, MultipleValuesOperation::IsBool);
    implement_assertion_operation!(is_datetime, MultipleValuesOperation::IsDateTime);
    implement_assertion_operation!(is_duration, MultipleValuesOperation::IsDuration);
    implement_assertion_operation!(is_null, MultipleValuesOperation::IsNull);
    implement_assertion_operation!(is_max, MultipleValuesOperation::IsMax);
    implement_assertion_operation!(is_min, MultipleValuesOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
    {
        let mut either_operand = Wrapper::<MultipleValuesOperand<O>>::new(self.context.clone());
        let mut or_operand = Wrapper::<MultipleValuesOperand<O>>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(MultipleValuesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
    {
        let mut operand = Wrapper::<MultipleValuesOperand<O>>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesOperation::Exclude { operand });
    }
}

impl<O: Operand> Wrapper<MultipleValuesOperand<O>> {
    pub(crate) fn new(context: Context<O>) -> Self {
        MultipleValuesOperand::new(context).into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (O::Index, MedRecordValue)> + 'a>
    where
        O: 'a,
    {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_return!(max, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(min, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(mean, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(median, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(mode, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(std, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(var, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(count, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(first, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(last, SingleValueOperand<O>);

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(ends_with, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleValuesComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleValuesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(div, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleValueComparisonOperand<O>>);

    implement_wrapper_operand!(round);
    implement_wrapper_operand!(ceil);
    implement_wrapper_operand!(floor);
    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(sqrt);
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
    implement_wrapper_operand!(is_float);
    implement_wrapper_operand!(is_bool);
    implement_wrapper_operand!(is_datetime);
    implement_wrapper_operand!(is_duration);
    implement_wrapper_operand!(is_null);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

#[derive(Debug, Clone)]
pub struct SingleValueOperand<O: Operand> {
    pub(crate) context: MultipleValuesOperand<O>,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleValueOperation<O>>,
}

impl<O: Operand> DeepClone for SingleValueOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<O: Operand> SingleValueOperand<O> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<O::Index, MedRecordValue>>>
    where
        O: 'a,
    {
        let values = self.context.evaluate(medrecord)?;

        let value = match self.kind {
            SingleKind::Max => {
                OptionalIndexWrapper::WithIndex(MultipleValuesOperation::<O>::get_max(values)?)
            }
            SingleKind::Min => {
                OptionalIndexWrapper::WithIndex(MultipleValuesOperation::<O>::get_min(values)?)
            }
            SingleKind::Mean => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_mean(values)?)
            }
            SingleKind::Median => OptionalIndexWrapper::WithoutIndex(
                MultipleValuesOperation::<O>::get_median(values)?,
            ),
            SingleKind::Mode => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_mode(values)?)
            }
            SingleKind::Std => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_std(values)?)
            }
            SingleKind::Var => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_var(values)?)
            }
            SingleKind::Count => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_count(values))
            }
            SingleKind::Sum => {
                OptionalIndexWrapper::WithoutIndex(MultipleValuesOperation::<O>::get_sum(values)?)
            }
            SingleKind::First => {
                OptionalIndexWrapper::WithIndex(MultipleValuesOperation::<O>::get_first(values)?)
            }
            SingleKind::Last => {
                OptionalIndexWrapper::WithIndex(MultipleValuesOperation::<O>::get_last(values)?)
            }
        };

        self.operations
            .iter()
            .try_fold(Some(value), |value, operation| {
                if let Some(value) = value {
                    operation.evaluate(medrecord, value)
                } else {
                    Ok(None)
                }
            })
    }

    implement_single_value_comparison_operation!(greater_than, SingleValueOperation, GreaterThan);
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        SingleValueOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, SingleValueOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        SingleValueOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, SingleValueOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, SingleValueOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, SingleValueOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, SingleValueOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, SingleValueOperation, Contains);

    pub fn is_in<V: Into<MultipleValuesComparisonOperand<O>>>(&mut self, values: V) {
        self.operations
            .push(SingleValueOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand<O>>>(&mut self, values: V) {
        self.operations
            .push(SingleValueOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, SingleValueOperation, Add);
    implement_binary_arithmetic_operation!(sub, SingleValueOperation, Sub);
    implement_binary_arithmetic_operation!(mul, SingleValueOperation, Mul);
    implement_binary_arithmetic_operation!(div, SingleValueOperation, Div);
    implement_binary_arithmetic_operation!(pow, SingleValueOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleValueOperation, Mod);

    implement_unary_arithmetic_operation!(round, SingleValueOperation, Round);
    implement_unary_arithmetic_operation!(ceil, SingleValueOperation, Ceil);
    implement_unary_arithmetic_operation!(floor, SingleValueOperation, Floor);
    implement_unary_arithmetic_operation!(abs, SingleValueOperation, Abs);
    implement_unary_arithmetic_operation!(sqrt, SingleValueOperation, Sqrt);
    implement_unary_arithmetic_operation!(trim, SingleValueOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, SingleValueOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, SingleValueOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, SingleValueOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, SingleValueOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleValueOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleValueOperation::IsString);
    implement_assertion_operation!(is_int, SingleValueOperation::IsInt);
    implement_assertion_operation!(is_float, SingleValueOperation::IsFloat);
    implement_assertion_operation!(is_bool, SingleValueOperation::IsBool);
    implement_assertion_operation!(is_datetime, SingleValueOperation::IsDateTime);
    implement_assertion_operation!(is_duration, SingleValueOperation::IsDuration);
    implement_assertion_operation!(is_null, SingleValueOperation::IsNull);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<SingleValueOperand<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleValueOperand<O>>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(SingleValueOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
    {
        let mut operand =
            Wrapper::<SingleValueOperand<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleValueOperation::Exclude { operand });
    }
}

impl<O: Operand> Wrapper<SingleValueOperand<O>> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKind) -> Self {
        SingleValueOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<O::Index, MedRecordValue>>>
    where
        O: 'a,
    {
        self.0.read_or_panic().evaluate(medrecord)
    }

    implement_wrapper_operand_with_argument!(
        greater_than,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        not_equal_to,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(
        starts_with,
        impl Into<SingleValueComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(ends_with, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleValuesComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(
        is_not_in,
        impl Into<MultipleValuesComparisonOperand<O>>
    );
    implement_wrapper_operand_with_argument!(add, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(div, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleValueComparisonOperand<O>>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleValueComparisonOperand<O>>);

    implement_wrapper_operand!(round);
    implement_wrapper_operand!(ceil);
    implement_wrapper_operand!(floor);
    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(sqrt);
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
    implement_wrapper_operand!(is_float);
    implement_wrapper_operand!(is_bool);
    implement_wrapper_operand!(is_datetime);
    implement_wrapper_operand!(is_duration);
    implement_wrapper_operand!(is_null);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

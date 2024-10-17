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
            BoxedIterator,
        },
        MedRecordAttribute, MedRecordValue, Wrapper,
    },
    MedRecord,
};
use std::hash::Hash;

macro_rules! implement_value_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueOperand> {
            let operand =
                Wrapper::<SingleValueOperand>::new(self.deep_clone(), SingleKind::$variant);

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
        pub fn $name<V: Into<SingleValueComparisonOperand>>(&mut self, value: V) {
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
        pub fn $name<V: Into<SingleValueComparisonOperand>>(&mut self, value: V) {
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
    ($name:ident, $return_operand:ident) => {
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
pub enum SingleValueComparisonOperand {
    Operand(SingleValueOperand),
    Value(MedRecordValue),
}

impl DeepClone for SingleValueComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Value(value) => Self::Value(value.clone()),
        }
    }
}

impl From<Wrapper<SingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<SingleValueOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<SingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<SingleValueOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>> From<V> for SingleValueComparisonOperand {
    fn from(value: V) -> Self {
        Self::Value(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesComparisonOperand {
    Operand(MultipleValuesOperand),
    Values(Vec<MedRecordValue>),
}

impl DeepClone for MultipleValuesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Values(value) => Self::Values(value.clone()),
        }
    }
}

impl From<Wrapper<MultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<MultipleValuesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<MultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<MultipleValuesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>> From<Vec<V>> for MultipleValuesComparisonOperand {
    fn from(value: Vec<V>) -> Self {
        Self::Values(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordValue> + Clone, const N: usize> From<[V; N]>
    for MultipleValuesComparisonOperand
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

#[derive(Debug, Clone)]
pub struct MultipleValuesOperand {
    pub(crate) context: Context,
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<MultipleValuesOperation>,
}

impl DeepClone for MultipleValuesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            attribute: self.attribute.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl MultipleValuesOperand {
    pub(crate) fn new(context: Context, attribute: MedRecordAttribute) -> Self {
        Self {
            context,
            attribute,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Hash>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, MedRecordValue)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (&'a T, MedRecordValue)>> {
        let values = Box::new(values) as BoxedIterator<(&'a T, MedRecordValue)>;

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

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(MultipleValuesOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
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
    implement_assertion_operation!(is_null, MultipleValuesOperation::IsNull);
    implement_assertion_operation!(is_max, MultipleValuesOperation::IsMax);
    implement_assertion_operation!(is_min, MultipleValuesOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesOperand>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesOperand>),
    {
        let mut either_operand =
            Wrapper::<MultipleValuesOperand>::new(self.context.clone(), self.attribute.clone());
        let mut or_operand =
            Wrapper::<MultipleValuesOperand>::new(self.context.clone(), self.attribute.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(MultipleValuesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesOperand>),
    {
        let mut operand =
            Wrapper::<MultipleValuesOperand>::new(self.context.clone(), self.attribute.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesOperation::Exclude { operand });
    }
}

impl Wrapper<MultipleValuesOperand> {
    pub(crate) fn new(context: Context, attribute: MedRecordAttribute) -> Self {
        MultipleValuesOperand::new(context, attribute).into()
    }

    pub(crate) fn evaluate<'a, T: 'a + Eq + Hash>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, MedRecordValue)> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = (&'a T, MedRecordValue)>> {
        self.0.read_or_panic().evaluate(medrecord, values)
    }

    implement_wrapper_operand_with_return!(max, SingleValueOperand);
    implement_wrapper_operand_with_return!(min, SingleValueOperand);
    implement_wrapper_operand_with_return!(mean, SingleValueOperand);
    implement_wrapper_operand_with_return!(median, SingleValueOperand);
    implement_wrapper_operand_with_return!(mode, SingleValueOperand);
    implement_wrapper_operand_with_return!(std, SingleValueOperand);
    implement_wrapper_operand_with_return!(var, SingleValueOperand);
    implement_wrapper_operand_with_return!(count, SingleValueOperand);
    implement_wrapper_operand_with_return!(sum, SingleValueOperand);
    implement_wrapper_operand_with_return!(first, SingleValueOperand);
    implement_wrapper_operand_with_return!(last, SingleValueOperand);

    implement_wrapper_operand_with_argument!(greater_than, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleValueComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleValueComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleValuesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<MultipleValuesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(div, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleValueComparisonOperand>);

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
    implement_wrapper_operand!(is_null);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesOperand>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

#[derive(Debug, Clone)]
pub struct SingleValueOperand {
    pub(crate) context: MultipleValuesOperand,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleValueOperation>,
}

impl DeepClone for SingleValueOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl SingleValueOperand {
    pub(crate) fn new(context: MultipleValuesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: MedRecordValue,
    ) -> MedRecordResult<Option<MedRecordValue>> {
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

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(SingleValueOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
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
    implement_assertion_operation!(is_null, SingleValueOperation::IsNull);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperand>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperand>),
    {
        let mut either_operand =
            Wrapper::<SingleValueOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleValueOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(SingleValueOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperand>),
    {
        let mut operand =
            Wrapper::<SingleValueOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleValueOperation::Exclude { operand });
    }
}

impl Wrapper<SingleValueOperand> {
    pub(crate) fn new(context: MultipleValuesOperand, kind: SingleKind) -> Self {
        SingleValueOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: MedRecordValue,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        self.0.read_or_panic().evaluate(medrecord, value)
    }

    implement_wrapper_operand_with_argument!(greater_than, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<SingleValueComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<SingleValueComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<MultipleValuesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<MultipleValuesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(div, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<SingleValueComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<SingleValueComparisonOperand>);

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
    implement_wrapper_operand!(is_null);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperand>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperand>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

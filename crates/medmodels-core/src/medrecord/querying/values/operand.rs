use super::{
    operation::{MultipleValuesOperation, SingleValueOperation},
    BinaryArithmeticKind, Context, MultipleComparisonKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            operand_traits::{Count, Max},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            OptionalIndexWrapper, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeOperand, MedRecordValue, NodeOperand, Wrapper,
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
pub enum SingleValueComparisonOperand {
    NodeSingleValueOperand(NodeSingleValueOperand),
    EdgeSingleValueOperand(EdgeSingleValueOperand),
    Value(MedRecordValue),
}

impl DeepClone for SingleValueComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeSingleValueOperand(operand) => {
                Self::NodeSingleValueOperand(operand.deep_clone())
            }
            Self::EdgeSingleValueOperand(operand) => {
                Self::EdgeSingleValueOperand(operand.deep_clone())
            }
            Self::Value(value) => Self::Value(value.clone()),
        }
    }
}

impl From<Wrapper<NodeSingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<NodeSingleValueOperand>) -> Self {
        Self::NodeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<NodeSingleValueOperand>) -> Self {
        Self::NodeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<EdgeSingleValueOperand>) -> Self {
        Self::EdgeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleValueOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleValueOperand>) -> Self {
        Self::EdgeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>> From<V> for SingleValueComparisonOperand {
    fn from(value: V) -> Self {
        Self::Value(value.into())
    }
}

impl SingleValueComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        Ok(match self {
            Self::NodeSingleValueOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.unpack().1),
            Self::EdgeSingleValueOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.unpack().1),
            Self::Value(value) => Some(value.clone()),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesComparisonOperand {
    NodeMultipleValuesOperand(NodeMultipleValuesOperand),
    EdgeMultipleValuesOperand(EdgeMultipleValuesOperand),
    Values(Vec<MedRecordValue>),
}

impl DeepClone for MultipleValuesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeMultipleValuesOperand(operand) => {
                Self::NodeMultipleValuesOperand(operand.deep_clone())
            }
            Self::EdgeMultipleValuesOperand(operand) => {
                Self::EdgeMultipleValuesOperand(operand.deep_clone())
            }
            Self::Values(value) => Self::Values(value.clone()),
        }
    }
}

impl From<Wrapper<NodeMultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleValuesOperand>) -> Self {
        Self::NodeMultipleValuesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<NodeMultipleValuesOperand>) -> Self {
        Self::NodeMultipleValuesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleValuesOperand>) -> Self {
        Self::EdgeMultipleValuesOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleValuesOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<EdgeMultipleValuesOperand>) -> Self {
        Self::EdgeMultipleValuesOperand(value.0.read_or_panic().deep_clone())
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

impl MultipleValuesComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Vec<MedRecordValue>> {
        Ok(match self {
            Self::NodeMultipleValuesOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::EdgeMultipleValuesOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::Values(values) => values.clone(),
        })
    }
}

pub type NodeMultipleValuesOperand = MultipleValuesOperand<NodeOperand>;
pub type EdgeMultipleValuesOperand = MultipleValuesOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleValuesOperand<O: RootOperand> {
    pub(crate) context: Context<O>,
    operations: Vec<MultipleValuesOperation<O>>,
}

impl<O: RootOperand> DeepClone for MultipleValuesOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleValuesOperand<O> {
    type InputValue = BoxedIterator<'a, (&'a O::Index, MedRecordValue)>;
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordValue)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        values: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let values = Box::new(values) as BoxedIterator<_>;

        self.operations
            .iter()
            .try_fold(values, |value_tuples, operation| {
                operation.evaluate(medrecord, value_tuples)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for MultipleValuesOperand<O> {
    fn evaluate_forward_grouped(
        &self,
        _medrecord: &'a MedRecord,
        _values: BoxedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>> {
        todo!()
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleValuesOperand<O> {
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordValue)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.get_values(medrecord)?;

        self.evaluate_forward(medrecord, Box::new(values))
    }
}

impl<O: RootOperand> Max for MultipleValuesOperand<O> {
    type ReturnOperand = SingleValueOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueOperand<O>>::new(self.deep_clone(), SingleKind::Max);

        self.operations
            .push(MultipleValuesOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for MultipleValuesOperand<O> {
    type ReturnOperand = SingleValueOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueOperand<O>>::new(self.deep_clone(), SingleKind::Count);

        self.operations
            .push(MultipleValuesOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> MultipleValuesOperand<O> {
    pub(crate) fn new(context: Context<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    implement_value_operation!(min, Min);
    implement_value_operation!(mean, Mean);
    implement_value_operation!(median, Median);
    implement_value_operation!(mode, Mode);
    implement_value_operation!(std, Std);
    implement_value_operation!(var, Var);
    implement_value_operation!(count, Count);
    implement_value_operation!(sum, Sum);
    implement_value_operation!(random, Random);

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

impl<O: RootOperand> Wrapper<MultipleValuesOperand<O>> {
    pub(crate) fn new(context: Context<O>) -> Self {
        MultipleValuesOperand::new(context).into()
    }

    implement_wrapper_operand_with_return!(min, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(mean, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(median, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(mode, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(std, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(var, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleValueOperand<O>);
    implement_wrapper_operand_with_return!(random, SingleValueOperand<O>);

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

pub type NodeSingleValueOperand = SingleValueOperand<NodeOperand>;
pub type EdgeSingleValueOperand = SingleValueOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueOperand<O: RootOperand> {
    context: MultipleValuesOperand<O>,
    pub(crate) kind: SingleKind,
    operations: Vec<SingleValueOperation<O>>,
}

impl<O: RootOperand> DeepClone for SingleValueOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleValueOperand<O> {
    type InputValue = OptionalIndexWrapper<&'a O::Index, MedRecordValue>;
    type ReturnValue = Option<OptionalIndexWrapper<&'a O::Index, MedRecordValue>>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        value: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
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
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleValueOperand<O> {
    type ReturnValue = Option<OptionalIndexWrapper<&'a O::Index, MedRecordValue>>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.evaluate_backward(medrecord)?;

        let value = self.reduce_input(values)?;

        self.evaluate_forward(medrecord, value)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleValueOperand<O> {
    type Context = MultipleValuesOperand<O>;

    #[inline]
    fn reduce_input(
        &self,
        values: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKind::Max => MultipleValuesOperation::<O>::get_max(values)?.into(),
            SingleKind::Min => MultipleValuesOperation::<O>::get_min(values)?.into(),
            SingleKind::Mean => MultipleValuesOperation::<O>::get_mean(values)?.into(),
            SingleKind::Median => MultipleValuesOperation::<O>::get_median(values)?.into(),
            SingleKind::Mode => MultipleValuesOperation::<O>::get_mode(values)?.into(),
            SingleKind::Std => MultipleValuesOperation::<O>::get_std(values)?.into(),
            SingleKind::Var => MultipleValuesOperation::<O>::get_var(values)?.into(),
            SingleKind::Count => MultipleValuesOperation::<O>::get_count(values).into(),
            SingleKind::Sum => MultipleValuesOperation::<O>::get_sum(values)?.into(),
            SingleKind::Random => MultipleValuesOperation::<O>::get_random(values)?.into(),
        })
    }
}

impl<O: RootOperand> SingleValueOperand<O> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
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

impl<O: RootOperand> Wrapper<SingleValueOperand<O>> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKind) -> Self {
        SingleValueOperand::new(context, kind).into()
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

use super::{
    operation::{MultipleValuesOperation, SingleValueOperationWithIndex},
    BinaryArithmeticKind, Context, MultipleComparisonKind, SingleComparisonKind,
    SingleKindWithIndex, UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            operand_traits::{Count, Max},
            values::{operation::SingleValueOperationWithoutIndex, SingleKindWithoutIndex},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeOperand, MedRecordValue, NodeOperand, Wrapper,
    },
    MedRecord,
};

macro_rules! implement_value_operation_with_index {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueOperandWithIndex<O>> {
            let operand = Wrapper::<SingleValueOperandWithIndex<O>>::new(
                self.deep_clone(),
                SingleKindWithIndex::$variant,
            );

            self.operations
                .push(MultipleValuesOperation::ValueOperationWithIndex {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_value_operation_without_index {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueOperandWithoutIndex<O>> {
            let operand = Wrapper::<SingleValueOperandWithoutIndex<O>>::new(
                self.deep_clone(),
                SingleKindWithoutIndex::$variant,
            );

            self.operations
                .push(MultipleValuesOperation::ValueOperationWithoutIndex {
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
    NodeSingleValueOperand(NodeSingleValueOperandWithIndex),
    EdgeSingleValueOperand(EdgeSingleValueOperandWithIndex),
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

impl From<Wrapper<NodeSingleValueOperandWithIndex>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<NodeSingleValueOperandWithIndex>) -> Self {
        Self::NodeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleValueOperandWithIndex>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<NodeSingleValueOperandWithIndex>) -> Self {
        Self::NodeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleValueOperandWithIndex>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<EdgeSingleValueOperandWithIndex>) -> Self {
        Self::EdgeSingleValueOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleValueOperandWithIndex>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleValueOperandWithIndex>) -> Self {
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
                .map(|attribute| attribute.1),
            Self::EdgeSingleValueOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
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
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(values, |value_tuples, operation| {
                operation.evaluate_grouped(medrecord, value_tuples)
            })
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
    type ReturnOperand = SingleValueOperandWithIndex<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueOperandWithIndex<O>>::new(
            self.deep_clone(),
            SingleKindWithIndex::Max,
        );

        self.operations
            .push(MultipleValuesOperation::ValueOperationWithIndex {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for MultipleValuesOperand<O> {
    type ReturnOperand = SingleValueOperandWithoutIndex<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueOperandWithoutIndex<O>>::new(
            self.deep_clone(),
            SingleKindWithoutIndex::Count,
        );

        self.operations
            .push(MultipleValuesOperation::ValueOperationWithoutIndex {
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

    implement_value_operation_with_index!(min, Min);
    implement_value_operation_without_index!(mean, Mean);
    implement_value_operation_without_index!(median, Median);
    implement_value_operation_without_index!(mode, Mode);
    implement_value_operation_without_index!(std, Std);
    implement_value_operation_without_index!(var, Var);
    implement_value_operation_without_index!(count, Count);
    implement_value_operation_without_index!(sum, Sum);
    implement_value_operation_with_index!(random, Random);

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

    implement_wrapper_operand_with_return!(min, SingleValueOperandWithIndex<O>);
    implement_wrapper_operand_with_return!(mean, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(median, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(mode, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(std, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(var, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(sum, SingleValueOperandWithoutIndex<O>);
    implement_wrapper_operand_with_return!(random, SingleValueOperandWithIndex<O>);

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

pub type NodeSingleValueOperandWithIndex = SingleValueOperandWithIndex<NodeOperand>;
pub type EdgeSingleValueOperandWithIndex = SingleValueOperandWithIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueOperandWithIndex<O: RootOperand> {
    context: MultipleValuesOperand<O>,
    pub(crate) kind: SingleKindWithIndex,
    operations: Vec<SingleValueOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleValueOperandWithIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleValueOperandWithIndex<O> {
    type InputValue = (&'a O::Index, MedRecordValue);
    type ReturnValue = Option<(&'a O::Index, MedRecordValue)>;

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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleValueOperandWithIndex<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        let values = Box::new(values.map(|(key, value)| (key, Some(value))))
            as GroupedIterator<'a, Self::ReturnValue>;

        self.operations
            .iter()
            .try_fold(values, |values, operation| {
                operation.evaluate_grouped(medrecord, values)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleValueOperandWithIndex<O> {
    type ReturnValue = Option<(&'a O::Index, MedRecordValue)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.evaluate_backward(medrecord)?;

        let value = self.reduce_input(values)?;

        self.evaluate_forward(medrecord, value)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleValueOperandWithIndex<O> {
    type Context = MultipleValuesOperand<O>;

    #[inline]
    fn reduce_input(
        &self,
        values: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKindWithIndex::Max => MultipleValuesOperation::<O>::get_max(values)?,
            SingleKindWithIndex::Min => MultipleValuesOperation::<O>::get_min(values)?,
            SingleKindWithIndex::Random => MultipleValuesOperation::<O>::get_random(values)?,
        })
    }
}

impl<O: RootOperand> SingleValueOperandWithIndex<O> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKindWithIndex) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_single_value_comparison_operation!(
        greater_than,
        SingleValueOperationWithIndex,
        GreaterThan
    );
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        SingleValueOperationWithIndex,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        less_than,
        SingleValueOperationWithIndex,
        LessThan
    );
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        SingleValueOperationWithIndex,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, SingleValueOperationWithIndex, EqualTo);
    implement_single_value_comparison_operation!(
        not_equal_to,
        SingleValueOperationWithIndex,
        NotEqualTo
    );
    implement_single_value_comparison_operation!(
        starts_with,
        SingleValueOperationWithIndex,
        StartsWith
    );
    implement_single_value_comparison_operation!(
        ends_with,
        SingleValueOperationWithIndex,
        EndsWith
    );
    implement_single_value_comparison_operation!(contains, SingleValueOperationWithIndex, Contains);

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueOperationWithIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueOperationWithIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, SingleValueOperationWithIndex, Add);
    implement_binary_arithmetic_operation!(sub, SingleValueOperationWithIndex, Sub);
    implement_binary_arithmetic_operation!(mul, SingleValueOperationWithIndex, Mul);
    implement_binary_arithmetic_operation!(div, SingleValueOperationWithIndex, Div);
    implement_binary_arithmetic_operation!(pow, SingleValueOperationWithIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleValueOperationWithIndex, Mod);

    implement_unary_arithmetic_operation!(round, SingleValueOperationWithIndex, Round);
    implement_unary_arithmetic_operation!(ceil, SingleValueOperationWithIndex, Ceil);
    implement_unary_arithmetic_operation!(floor, SingleValueOperationWithIndex, Floor);
    implement_unary_arithmetic_operation!(abs, SingleValueOperationWithIndex, Abs);
    implement_unary_arithmetic_operation!(sqrt, SingleValueOperationWithIndex, Sqrt);
    implement_unary_arithmetic_operation!(trim, SingleValueOperationWithIndex, Trim);
    implement_unary_arithmetic_operation!(trim_start, SingleValueOperationWithIndex, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, SingleValueOperationWithIndex, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, SingleValueOperationWithIndex, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, SingleValueOperationWithIndex, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleValueOperationWithIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleValueOperationWithIndex::IsString);
    implement_assertion_operation!(is_int, SingleValueOperationWithIndex::IsInt);
    implement_assertion_operation!(is_float, SingleValueOperationWithIndex::IsFloat);
    implement_assertion_operation!(is_bool, SingleValueOperationWithIndex::IsBool);
    implement_assertion_operation!(is_datetime, SingleValueOperationWithIndex::IsDateTime);
    implement_assertion_operation!(is_duration, SingleValueOperationWithIndex::IsDuration);
    implement_assertion_operation!(is_null, SingleValueOperationWithIndex::IsNull);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
    {
        let mut either_operand =
            Wrapper::<SingleValueOperandWithIndex<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleValueOperandWithIndex<O>>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleValueOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
    {
        let mut operand =
            Wrapper::<SingleValueOperandWithIndex<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleValueOperationWithIndex::Exclude { operand });
    }

    pub(crate) fn push_merge_operation(&mut self, operand: Wrapper<MultipleValuesOperand<O>>) {
        self.operations
            .push(SingleValueOperationWithIndex::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueOperandWithIndex<O>> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKindWithIndex) -> Self {
        SingleValueOperandWithIndex::new(context, kind).into()
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
        EQ: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperandWithIndex<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }

    pub(crate) fn push_merge_operation(&self, operand: Wrapper<MultipleValuesOperand<O>>) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

pub type NodeSingleValueOperandWithoutIndex = SingleValueOperandWithoutIndex<NodeOperand>;
pub type EdgeSingleValueOperandWithoutIndex = SingleValueOperandWithoutIndex<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueOperandWithoutIndex<O: RootOperand> {
    context: MultipleValuesOperand<O>,
    pub(crate) kind: SingleKindWithoutIndex,
    operations: Vec<SingleValueOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleValueOperandWithoutIndex<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleValueOperandWithoutIndex<O> {
    type InputValue = MedRecordValue;
    type ReturnValue = Option<MedRecordValue>;

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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleValueOperandWithoutIndex<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        let values = Box::new(values.map(|(key, value)| (key, Some(value))))
            as GroupedIterator<'a, Self::ReturnValue>;

        self.operations
            .iter()
            .try_fold(values, |values, operation| {
                operation.evaluate_grouped(medrecord, values)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleValueOperandWithoutIndex<O> {
    type ReturnValue = Option<MedRecordValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.evaluate_backward(medrecord)?;

        let value = self.reduce_input(values)?;

        self.evaluate_forward(medrecord, value)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleValueOperandWithoutIndex<O> {
    type Context = MultipleValuesOperand<O>;

    #[inline]
    fn reduce_input(
        &self,
        values: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKindWithoutIndex::Mean => MultipleValuesOperation::<O>::get_mean(values)?,
            SingleKindWithoutIndex::Median => MultipleValuesOperation::<O>::get_median(values)?,
            SingleKindWithoutIndex::Mode => MultipleValuesOperation::<O>::get_mode(values)?,
            SingleKindWithoutIndex::Std => MultipleValuesOperation::<O>::get_std(values)?,
            SingleKindWithoutIndex::Var => MultipleValuesOperation::<O>::get_var(values)?,
            SingleKindWithoutIndex::Count => MultipleValuesOperation::<O>::get_count(values),
            SingleKindWithoutIndex::Sum => MultipleValuesOperation::<O>::get_sum(values)?,
        })
    }
}

impl<O: RootOperand> SingleValueOperandWithoutIndex<O> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKindWithoutIndex) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    implement_single_value_comparison_operation!(
        greater_than,
        SingleValueOperationWithoutIndex,
        GreaterThan
    );
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        SingleValueOperationWithoutIndex,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        less_than,
        SingleValueOperationWithoutIndex,
        LessThan
    );
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        SingleValueOperationWithoutIndex,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        equal_to,
        SingleValueOperationWithoutIndex,
        EqualTo
    );
    implement_single_value_comparison_operation!(
        not_equal_to,
        SingleValueOperationWithoutIndex,
        NotEqualTo
    );
    implement_single_value_comparison_operation!(
        starts_with,
        SingleValueOperationWithoutIndex,
        StartsWith
    );
    implement_single_value_comparison_operation!(
        ends_with,
        SingleValueOperationWithoutIndex,
        EndsWith
    );
    implement_single_value_comparison_operation!(
        contains,
        SingleValueOperationWithoutIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueOperationWithoutIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueOperationWithoutIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, SingleValueOperationWithoutIndex, Add);
    implement_binary_arithmetic_operation!(sub, SingleValueOperationWithoutIndex, Sub);
    implement_binary_arithmetic_operation!(mul, SingleValueOperationWithoutIndex, Mul);
    implement_binary_arithmetic_operation!(div, SingleValueOperationWithoutIndex, Div);
    implement_binary_arithmetic_operation!(pow, SingleValueOperationWithoutIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, SingleValueOperationWithoutIndex, Mod);

    implement_unary_arithmetic_operation!(round, SingleValueOperationWithoutIndex, Round);
    implement_unary_arithmetic_operation!(ceil, SingleValueOperationWithoutIndex, Ceil);
    implement_unary_arithmetic_operation!(floor, SingleValueOperationWithoutIndex, Floor);
    implement_unary_arithmetic_operation!(abs, SingleValueOperationWithoutIndex, Abs);
    implement_unary_arithmetic_operation!(sqrt, SingleValueOperationWithoutIndex, Sqrt);
    implement_unary_arithmetic_operation!(trim, SingleValueOperationWithoutIndex, Trim);
    implement_unary_arithmetic_operation!(trim_start, SingleValueOperationWithoutIndex, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, SingleValueOperationWithoutIndex, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, SingleValueOperationWithoutIndex, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, SingleValueOperationWithoutIndex, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleValueOperationWithoutIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, SingleValueOperationWithoutIndex::IsString);
    implement_assertion_operation!(is_int, SingleValueOperationWithoutIndex::IsInt);
    implement_assertion_operation!(is_float, SingleValueOperationWithoutIndex::IsFloat);
    implement_assertion_operation!(is_bool, SingleValueOperationWithoutIndex::IsBool);
    implement_assertion_operation!(is_datetime, SingleValueOperationWithoutIndex::IsDateTime);
    implement_assertion_operation!(is_duration, SingleValueOperationWithoutIndex::IsDuration);
    implement_assertion_operation!(is_null, SingleValueOperationWithoutIndex::IsNull);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
    {
        let mut either_operand = Wrapper::<SingleValueOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<SingleValueOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleValueOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
    {
        let mut operand = Wrapper::<SingleValueOperandWithoutIndex<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(SingleValueOperationWithoutIndex::Exclude { operand });
    }

    pub(crate) fn _push_merge_operation(&mut self, operand: Wrapper<MultipleValuesOperand<O>>) {
        self.operations
            .push(SingleValueOperationWithoutIndex::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueOperandWithoutIndex<O>> {
    pub(crate) fn new(context: MultipleValuesOperand<O>, kind: SingleKindWithoutIndex) -> Self {
        SingleValueOperandWithoutIndex::new(context, kind).into()
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
        EQ: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueOperandWithoutIndex<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }

    pub(crate) fn _push_merge_operation(&self, operand: Wrapper<MultipleValuesOperand<O>>) {
        self.0.write_or_panic()._push_merge_operation(operand);
    }
}

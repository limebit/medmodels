use super::{
    operation::{MultipleValuesOperationWithIndex, SingleValueOperationWithIndex},
    BinaryArithmeticKind, MultipleComparisonKind, MultipleValuesWithIndexContext,
    SingleComparisonKind, SingleKindWithIndex, UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            operand_traits::{Count, Max},
            values::{
                operation::{
                    MultipleValuesOperationWithoutIndex, SingleValueOperationWithoutIndex,
                },
                MultipleValuesWithoutIndexContext, SingleKindWithoutIndex,
                SingleValueWithoutIndexContext,
            },
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeOperand, MedRecordValue, NodeOperand, Wrapper,
    },
    MedRecord,
};

macro_rules! implement_value_operation_with_index {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueWithIndexOperand<O>> {
            let operand = Wrapper::<SingleValueWithIndexOperand<O>>::new(
                self.deep_clone(),
                SingleKindWithIndex::$variant,
            );

            self.operations
                .push(MultipleValuesOperationWithIndex::ValueOperationWithIndex {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_value_operation_without_index {
    ($name:ident, $variant:ident, WithIndex) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueWithoutIndexOperand<O>> {
            let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
                SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
                SingleKindWithoutIndex::$variant,
            );

            self.operations.push(
                MultipleValuesOperationWithIndex::ValueOperationWithoutIndex {
                    operand: operand.clone(),
                },
            );

            operand
        }
    };
    ($name:ident, $variant:ident, WithoutIndex) => {
        pub fn $name(&mut self) -> Wrapper<SingleValueWithoutIndexOperand<O>> {
            let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
                SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(
                    self.deep_clone(),
                ),
                SingleKindWithoutIndex::$variant,
            );

            self.operations
                .push(MultipleValuesOperationWithoutIndex::ValueOperation {
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
    NodeSingleValueWithIndexOperand(NodeSingleValueWithIndexOperand),
    NodeSingleValueWithoutIndexOperand(NodeSingleValueWithoutIndexOperand),
    EdgeSingleValueWithIndexOperand(EdgeSingleValueWithIndexOperand),
    EdgeSingleValueWithoutIndexOperand(EdgeSingleValueWithoutIndexOperand),
    Value(MedRecordValue),
}

impl DeepClone for SingleValueComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeSingleValueWithIndexOperand(operand) => {
                Self::NodeSingleValueWithIndexOperand(operand.deep_clone())
            }
            Self::NodeSingleValueWithoutIndexOperand(operand) => {
                Self::NodeSingleValueWithoutIndexOperand(operand.deep_clone())
            }
            Self::EdgeSingleValueWithIndexOperand(operand) => {
                Self::EdgeSingleValueWithIndexOperand(operand.deep_clone())
            }
            Self::EdgeSingleValueWithoutIndexOperand(operand) => {
                Self::EdgeSingleValueWithoutIndexOperand(operand.deep_clone())
            }
            Self::Value(value) => Self::Value(value.clone()),
        }
    }
}

impl From<Wrapper<NodeSingleValueWithIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<NodeSingleValueWithIndexOperand>) -> Self {
        Self::NodeSingleValueWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleValueWithIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<NodeSingleValueWithIndexOperand>) -> Self {
        Self::NodeSingleValueWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<NodeSingleValueWithoutIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<NodeSingleValueWithoutIndexOperand>) -> Self {
        Self::NodeSingleValueWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleValueWithoutIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<NodeSingleValueWithoutIndexOperand>) -> Self {
        Self::NodeSingleValueWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleValueWithIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<EdgeSingleValueWithIndexOperand>) -> Self {
        Self::EdgeSingleValueWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleValueWithIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleValueWithIndexOperand>) -> Self {
        Self::EdgeSingleValueWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleValueWithoutIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: Wrapper<EdgeSingleValueWithoutIndexOperand>) -> Self {
        Self::EdgeSingleValueWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleValueWithoutIndexOperand>> for SingleValueComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleValueWithoutIndexOperand>) -> Self {
        Self::EdgeSingleValueWithoutIndexOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeSingleValueWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::NodeSingleValueWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::EdgeSingleValueWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::EdgeSingleValueWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::Value(value) => Some(value.clone()),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesComparisonOperand {
    NodeMultipleValuesWithIndexOperand(NodeMultipleValuesWithIndexOperand),
    NodeMultipleValuesWithoutIndexOperand(NodeMultipleValuesWithoutIndexOperand),
    EdgeMultipleValuesWithIndexOperand(EdgeMultipleValuesWithIndexOperand),
    EdgeMultipleValuesWithoutIndexOperand(EdgeMultipleValuesWithoutIndexOperand),
    Values(Vec<MedRecordValue>),
}

impl DeepClone for MultipleValuesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeMultipleValuesWithIndexOperand(operand) => {
                Self::NodeMultipleValuesWithIndexOperand(operand.deep_clone())
            }
            Self::NodeMultipleValuesWithoutIndexOperand(operand) => {
                Self::NodeMultipleValuesWithoutIndexOperand(operand.deep_clone())
            }
            Self::EdgeMultipleValuesWithIndexOperand(operand) => {
                Self::EdgeMultipleValuesWithIndexOperand(operand.deep_clone())
            }
            Self::EdgeMultipleValuesWithoutIndexOperand(operand) => {
                Self::EdgeMultipleValuesWithoutIndexOperand(operand.deep_clone())
            }
            Self::Values(value) => Self::Values(value.clone()),
        }
    }
}

impl From<Wrapper<NodeMultipleValuesWithIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleValuesWithIndexOperand>) -> Self {
        Self::NodeMultipleValuesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleValuesWithIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<NodeMultipleValuesWithIndexOperand>) -> Self {
        Self::NodeMultipleValuesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<NodeMultipleValuesWithoutIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleValuesWithoutIndexOperand>) -> Self {
        Self::NodeMultipleValuesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleValuesWithoutIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<NodeMultipleValuesWithoutIndexOperand>) -> Self {
        Self::NodeMultipleValuesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleValuesWithIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleValuesWithIndexOperand>) -> Self {
        Self::EdgeMultipleValuesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleValuesWithIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<EdgeMultipleValuesWithIndexOperand>) -> Self {
        Self::EdgeMultipleValuesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleValuesWithoutIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleValuesWithoutIndexOperand>) -> Self {
        Self::EdgeMultipleValuesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleValuesWithoutIndexOperand>> for MultipleValuesComparisonOperand {
    fn from(value: &Wrapper<EdgeMultipleValuesWithoutIndexOperand>) -> Self {
        Self::EdgeMultipleValuesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeMultipleValuesWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, value)| value)
                .collect(),
            Self::NodeMultipleValuesWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?.collect()
            }
            Self::EdgeMultipleValuesWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, value)| value)
                .collect(),
            Self::EdgeMultipleValuesWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?.collect()
            }
            Self::Values(values) => values.clone(),
        })
    }
}

pub type NodeMultipleValuesWithIndexOperand = MultipleValuesWithIndexOperand<NodeOperand>;
pub type EdgeMultipleValuesWithIndexOperand = MultipleValuesWithIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleValuesWithIndexOperand<O: RootOperand> {
    pub(crate) context: MultipleValuesWithIndexContext<O>,
    operations: Vec<MultipleValuesOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleValuesWithIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleValuesWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for MultipleValuesWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleValuesWithIndexOperand<O> {
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordValue)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.get_values(medrecord)?;

        self.evaluate_forward(medrecord, Box::new(values))
    }
}

impl<O: RootOperand> Max for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithIndexOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithIndexOperand<O>>::new(
            self.deep_clone(),
            SingleKindWithIndex::Max,
        );

        self.operations
            .push(MultipleValuesOperationWithIndex::ValueOperationWithIndex {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Count,
        );

        self.operations.push(
            MultipleValuesOperationWithIndex::ValueOperationWithoutIndex {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> MultipleValuesWithIndexOperand<O> {
    pub(crate) fn new(context: MultipleValuesWithIndexContext<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    implement_value_operation_with_index!(min, Min);
    implement_value_operation_without_index!(mean, Mean, WithIndex);
    implement_value_operation_without_index!(median, Median, WithIndex);
    implement_value_operation_without_index!(mode, Mode, WithIndex);
    implement_value_operation_without_index!(std, Std, WithIndex);
    implement_value_operation_without_index!(var, Var, WithIndex);
    implement_value_operation_without_index!(count, Count, WithIndex);
    implement_value_operation_without_index!(sum, Sum, WithIndex);
    implement_value_operation_with_index!(random, Random);

    implement_single_value_comparison_operation!(
        greater_than,
        MultipleValuesOperationWithIndex,
        GreaterThan
    );
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        MultipleValuesOperationWithIndex,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        less_than,
        MultipleValuesOperationWithIndex,
        LessThan
    );
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        MultipleValuesOperationWithIndex,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        equal_to,
        MultipleValuesOperationWithIndex,
        EqualTo
    );
    implement_single_value_comparison_operation!(
        not_equal_to,
        MultipleValuesOperationWithIndex,
        NotEqualTo
    );
    implement_single_value_comparison_operation!(
        starts_with,
        MultipleValuesOperationWithIndex,
        StartsWith
    );
    implement_single_value_comparison_operation!(
        ends_with,
        MultipleValuesOperationWithIndex,
        EndsWith
    );
    implement_single_value_comparison_operation!(
        contains,
        MultipleValuesOperationWithIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesOperationWithIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesOperationWithIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, MultipleValuesOperationWithIndex, Add);
    implement_binary_arithmetic_operation!(sub, MultipleValuesOperationWithIndex, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleValuesOperationWithIndex, Mul);
    implement_binary_arithmetic_operation!(div, MultipleValuesOperationWithIndex, Div);
    implement_binary_arithmetic_operation!(pow, MultipleValuesOperationWithIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleValuesOperationWithIndex, Mod);

    implement_unary_arithmetic_operation!(round, MultipleValuesOperationWithIndex, Round);
    implement_unary_arithmetic_operation!(ceil, MultipleValuesOperationWithIndex, Ceil);
    implement_unary_arithmetic_operation!(floor, MultipleValuesOperationWithIndex, Floor);
    implement_unary_arithmetic_operation!(abs, MultipleValuesOperationWithIndex, Abs);
    implement_unary_arithmetic_operation!(sqrt, MultipleValuesOperationWithIndex, Sqrt);
    implement_unary_arithmetic_operation!(trim, MultipleValuesOperationWithIndex, Trim);
    implement_unary_arithmetic_operation!(trim_start, MultipleValuesOperationWithIndex, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, MultipleValuesOperationWithIndex, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, MultipleValuesOperationWithIndex, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, MultipleValuesOperationWithIndex, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleValuesOperationWithIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleValuesOperationWithIndex::IsString);
    implement_assertion_operation!(is_int, MultipleValuesOperationWithIndex::IsInt);
    implement_assertion_operation!(is_float, MultipleValuesOperationWithIndex::IsFloat);
    implement_assertion_operation!(is_bool, MultipleValuesOperationWithIndex::IsBool);
    implement_assertion_operation!(is_datetime, MultipleValuesOperationWithIndex::IsDateTime);
    implement_assertion_operation!(is_duration, MultipleValuesOperationWithIndex::IsDuration);
    implement_assertion_operation!(is_null, MultipleValuesOperationWithIndex::IsNull);
    implement_assertion_operation!(is_max, MultipleValuesOperationWithIndex::IsMax);
    implement_assertion_operation!(is_min, MultipleValuesOperationWithIndex::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<MultipleValuesWithIndexOperand<O>>::new(self.context.clone());
        let mut or_operand =
            Wrapper::<MultipleValuesWithIndexOperand<O>>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleValuesOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
    {
        let mut operand = Wrapper::<MultipleValuesWithIndexOperand<O>>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesOperationWithIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<MultipleValuesWithIndexOperand<O>> {
    pub(crate) fn new(context: MultipleValuesWithIndexContext<O>) -> Self {
        MultipleValuesWithIndexOperand::new(context).into()
    }

    implement_wrapper_operand_with_return!(min, SingleValueWithIndexOperand<O>);
    implement_wrapper_operand_with_return!(mean, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(median, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(mode, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(std, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(var, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(random, SingleValueWithIndexOperand<O>);

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
        EQ: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesWithIndexOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

pub type NodeMultipleValuesWithoutIndexOperand = MultipleValuesWithoutIndexOperand<NodeOperand>;
pub type EdgeMultipleValuesWithoutIndexOperand = MultipleValuesWithoutIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleValuesWithoutIndexOperand<O: RootOperand> {
    pub(crate) context: MultipleValuesWithoutIndexContext<O>,
    operations: Vec<MultipleValuesOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleValuesWithoutIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleValuesWithoutIndexOperand<O> {
    type InputValue = BoxedIterator<'a, MedRecordValue>;
    type ReturnValue = BoxedIterator<'a, MedRecordValue>;

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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for MultipleValuesWithoutIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleValuesWithoutIndexOperand<O> {
    type ReturnValue = BoxedIterator<'a, MedRecordValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.get_values(medrecord)?;

        self.evaluate_forward(medrecord, Box::new(values))
    }
}

impl<O: RootOperand> Max for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Max,
        );

        self.operations
            .push(MultipleValuesOperationWithoutIndex::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Count,
        );

        self.operations
            .push(MultipleValuesOperationWithoutIndex::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> MultipleValuesWithoutIndexOperand<O> {
    pub(crate) fn new(context: MultipleValuesWithoutIndexContext<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    implement_value_operation_without_index!(min, Min, WithoutIndex);
    implement_value_operation_without_index!(mean, Mean, WithoutIndex);
    implement_value_operation_without_index!(median, Median, WithoutIndex);
    implement_value_operation_without_index!(mode, Mode, WithoutIndex);
    implement_value_operation_without_index!(std, Std, WithoutIndex);
    implement_value_operation_without_index!(var, Var, WithoutIndex);
    implement_value_operation_without_index!(count, Count, WithoutIndex);
    implement_value_operation_without_index!(sum, Sum, WithoutIndex);
    implement_value_operation_without_index!(random, Random, WithoutIndex);

    implement_single_value_comparison_operation!(
        greater_than,
        MultipleValuesOperationWithoutIndex,
        GreaterThan
    );
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        MultipleValuesOperationWithoutIndex,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        less_than,
        MultipleValuesOperationWithoutIndex,
        LessThan
    );
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        MultipleValuesOperationWithoutIndex,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(
        equal_to,
        MultipleValuesOperationWithoutIndex,
        EqualTo
    );
    implement_single_value_comparison_operation!(
        not_equal_to,
        MultipleValuesOperationWithoutIndex,
        NotEqualTo
    );
    implement_single_value_comparison_operation!(
        starts_with,
        MultipleValuesOperationWithoutIndex,
        StartsWith
    );
    implement_single_value_comparison_operation!(
        ends_with,
        MultipleValuesOperationWithoutIndex,
        EndsWith
    );
    implement_single_value_comparison_operation!(
        contains,
        MultipleValuesOperationWithoutIndex,
        Contains
    );

    pub fn is_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesOperationWithoutIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }

    pub fn is_not_in<V: Into<MultipleValuesComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesOperationWithoutIndex::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }

    implement_binary_arithmetic_operation!(add, MultipleValuesOperationWithoutIndex, Add);
    implement_binary_arithmetic_operation!(sub, MultipleValuesOperationWithoutIndex, Sub);
    implement_binary_arithmetic_operation!(mul, MultipleValuesOperationWithoutIndex, Mul);
    implement_binary_arithmetic_operation!(div, MultipleValuesOperationWithoutIndex, Div);
    implement_binary_arithmetic_operation!(pow, MultipleValuesOperationWithoutIndex, Pow);
    implement_binary_arithmetic_operation!(r#mod, MultipleValuesOperationWithoutIndex, Mod);

    implement_unary_arithmetic_operation!(round, MultipleValuesOperationWithoutIndex, Round);
    implement_unary_arithmetic_operation!(ceil, MultipleValuesOperationWithoutIndex, Ceil);
    implement_unary_arithmetic_operation!(floor, MultipleValuesOperationWithoutIndex, Floor);
    implement_unary_arithmetic_operation!(abs, MultipleValuesOperationWithoutIndex, Abs);
    implement_unary_arithmetic_operation!(sqrt, MultipleValuesOperationWithoutIndex, Sqrt);
    implement_unary_arithmetic_operation!(trim, MultipleValuesOperationWithoutIndex, Trim);
    implement_unary_arithmetic_operation!(
        trim_start,
        MultipleValuesOperationWithoutIndex,
        TrimStart
    );
    implement_unary_arithmetic_operation!(trim_end, MultipleValuesOperationWithoutIndex, TrimEnd);
    implement_unary_arithmetic_operation!(
        lowercase,
        MultipleValuesOperationWithoutIndex,
        Lowercase
    );
    implement_unary_arithmetic_operation!(
        uppercase,
        MultipleValuesOperationWithoutIndex,
        Uppercase
    );

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleValuesOperationWithoutIndex::Slice(start..end));
    }

    implement_assertion_operation!(is_string, MultipleValuesOperationWithoutIndex::IsString);
    implement_assertion_operation!(is_int, MultipleValuesOperationWithoutIndex::IsInt);
    implement_assertion_operation!(is_float, MultipleValuesOperationWithoutIndex::IsFloat);
    implement_assertion_operation!(is_bool, MultipleValuesOperationWithoutIndex::IsBool);
    implement_assertion_operation!(is_datetime, MultipleValuesOperationWithoutIndex::IsDateTime);
    implement_assertion_operation!(is_duration, MultipleValuesOperationWithoutIndex::IsDuration);
    implement_assertion_operation!(is_null, MultipleValuesOperationWithoutIndex::IsNull);
    implement_assertion_operation!(is_max, MultipleValuesOperationWithoutIndex::IsMax);
    implement_assertion_operation!(is_min, MultipleValuesOperationWithoutIndex::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<MultipleValuesWithoutIndexOperand<O>>::new(self.context.clone());
        let mut or_operand =
            Wrapper::<MultipleValuesWithoutIndexOperand<O>>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleValuesOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
    {
        let mut operand =
            Wrapper::<MultipleValuesWithoutIndexOperand<O>>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesOperationWithoutIndex::Exclude { operand });
    }
}

impl<O: RootOperand> Wrapper<MultipleValuesWithoutIndexOperand<O>> {
    pub(crate) fn new(context: MultipleValuesWithoutIndexContext<O>) -> Self {
        MultipleValuesWithoutIndexOperand::new(context).into()
    }

    implement_wrapper_operand_with_return!(min, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(mean, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(median, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(mode, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(std, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(var, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(sum, SingleValueWithoutIndexOperand<O>);
    implement_wrapper_operand_with_return!(random, SingleValueWithoutIndexOperand<O>);

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
        EQ: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<MultipleValuesWithoutIndexOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

pub type NodeSingleValueWithIndexOperand = SingleValueWithIndexOperand<NodeOperand>;
pub type EdgeSingleValueWithIndexOperand = SingleValueWithIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueWithIndexOperand<O: RootOperand> {
    context: MultipleValuesWithIndexOperand<O>,
    pub(crate) kind: SingleKindWithIndex,
    operations: Vec<SingleValueOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleValueWithIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleValueWithIndexOperand<O> {
    type InputValue = Option<(&'a O::Index, MedRecordValue)>;
    type ReturnValue = Option<(&'a O::Index, MedRecordValue)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        value: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations.iter().try_fold(value, |value, operation| {
            operation.evaluate(medrecord, value)
        })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleValueWithIndexOperand<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(values, |values, operation| {
                operation.evaluate_grouped(medrecord, values)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleValueWithIndexOperand<O> {
    type ReturnValue = Option<(&'a O::Index, MedRecordValue)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.evaluate_backward(medrecord)?;

        let value = self.reduce_input(values)?;

        self.evaluate_forward(medrecord, value)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleValueWithIndexOperand<O> {
    type Context = MultipleValuesWithIndexOperand<O>;

    #[inline]
    fn reduce_input(
        &self,
        values: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKindWithIndex::Max => MultipleValuesOperationWithIndex::<O>::get_max(values)?,
            SingleKindWithIndex::Min => MultipleValuesOperationWithIndex::<O>::get_min(values)?,
            SingleKindWithIndex::Random => {
                MultipleValuesOperationWithIndex::<O>::get_random(values)
            }
        })
    }
}

impl<O: RootOperand> SingleValueWithIndexOperand<O> {
    pub(crate) fn new(
        context: MultipleValuesWithIndexOperand<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
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
        EQ: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
    {
        let mut either_operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());

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
        Q: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
    {
        let mut operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleValueOperationWithIndex::Exclude { operand });
    }

    pub(crate) fn push_merge_operation(
        &mut self,
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) {
        self.operations
            .push(SingleValueOperationWithIndex::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueWithIndexOperand<O>> {
    pub(crate) fn new(
        context: MultipleValuesWithIndexOperand<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        SingleValueWithIndexOperand::new(context, kind).into()
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
        EQ: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueWithIndexOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }

    pub(crate) fn push_merge_operation(&self, operand: Wrapper<MultipleValuesWithIndexOperand<O>>) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

pub type NodeSingleValueWithoutIndexOperand = SingleValueWithoutIndexOperand<NodeOperand>;
pub type EdgeSingleValueWithoutIndexOperand = SingleValueWithoutIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueWithoutIndexOperand<O: RootOperand> {
    context: SingleValueWithoutIndexContext<O>,
    pub(crate) kind: SingleKindWithoutIndex,
    operations: Vec<SingleValueOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleValueWithoutIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleValueWithoutIndexOperand<O> {
    type InputValue = Option<MedRecordValue>;
    type ReturnValue = Option<MedRecordValue>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        value: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations.iter().try_fold(value, |value, operation| {
            operation.evaluate(medrecord, value)
        })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleValueWithoutIndexOperand<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(values, |values, operation| {
                operation.evaluate_grouped(medrecord, values)
            })
    }
}

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleValueWithoutIndexOperand<O> {
    type ReturnValue = Option<MedRecordValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.context.get_values(medrecord)?;

        let value = match self.kind {
            SingleKindWithoutIndex::Max => {
                MultipleValuesOperationWithoutIndex::<O>::get_max(values)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesOperationWithoutIndex::<O>::get_min(values)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesOperationWithoutIndex::<O>::get_mean(values)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesOperationWithoutIndex::<O>::get_median(values)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesOperationWithoutIndex::<O>::get_mode(values)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesOperationWithoutIndex::<O>::get_std(values)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesOperationWithoutIndex::<O>::get_var(values)?
            }
            SingleKindWithoutIndex::Count => {
                Some(MultipleValuesOperationWithoutIndex::<O>::get_count(values))
            }
            SingleKindWithoutIndex::Sum => {
                MultipleValuesOperationWithoutIndex::<O>::get_sum(values)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesOperationWithoutIndex::<O>::get_random(values)
            }
        };

        self.evaluate_forward(medrecord, value)
    }
}

impl<O: RootOperand> SingleValueWithoutIndexOperand<O> {
    pub(crate) fn new(
        context: SingleValueWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
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
        EQ: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
    {
        let mut either_operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );
        let mut or_operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
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
        Q: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
    {
        let mut operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(SingleValueOperationWithoutIndex::Exclude { operand });
    }

    pub(crate) fn push_merge_operation(
        &mut self,
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) {
        self.operations
            .push(SingleValueOperationWithoutIndex::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueWithoutIndexOperand<O>> {
    pub(crate) fn new(
        context: SingleValueWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
        SingleValueWithoutIndexOperand::new(context, kind).into()
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
        EQ: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
        OQ: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<SingleValueWithoutIndexOperand<O>>),
    {
        self.0.write_or_panic().exclude(query);
    }

    pub(crate) fn push_merge_operation(
        &self,
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

use super::{
    operation::{MultipleValuesWithIndexOperation, SingleValueWithIndexOperation},
    BinaryArithmeticKind, MultipleComparisonKind, MultipleValuesWithIndexContext,
    SingleComparisonKind, SingleKindWithIndex, UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            operand_traits::{
                Abs, Add, Ceil, Contains, Count, Div, EitherOr, EndsWith, EqualTo, Exclude, Floor,
                GreaterThan, GreaterThanOrEqualTo, IsBool, IsDateTime, IsDuration, IsFloat, IsIn,
                IsInt, IsMax, IsMin, IsNotIn, IsNull, IsString, LessThan, LessThanOrEqualTo,
                Lowercase, Max, Mean, Median, Min, Mod, Mode, Mul, NotEqualTo, Pow, Random, Round,
                Slice, Sqrt, StartsWith, Std, Sub, Sum, Trim, TrimEnd, TrimStart, Uppercase, Var,
            },
            values::{
                operation::{
                    MultipleValuesWithoutIndexOperation, SingleValueWithoutIndexOperation,
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
    operations: Vec<MultipleValuesWithIndexOperation<O>>,
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
        let values: BoxedIterator<_> = Box::new(values);

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
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Max);

        self.operations
            .push(MultipleValuesWithIndexOperation::ValueWithIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Min for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithIndexOperand<O>;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Min);

        self.operations
            .push(MultipleValuesWithIndexOperation::ValueWithIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Mean for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn mean(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Mean,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Median for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn median(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Median,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Mode for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn mode(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Mode,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Std for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn std(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Std,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Var for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn var(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Var,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Count for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Count,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Sum for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Sum,
        );

        self.operations.push(
            MultipleValuesWithIndexOperation::ValueWithoutIndexOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Random for MultipleValuesWithIndexOperand<O> {
    type ReturnOperand = SingleValueWithIndexOperand<O>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Random);

        self.operations
            .push(MultipleValuesWithIndexOperation::ValueWithIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> GreaterThan for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Div for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Div,
            },
        );
    }
}

impl<O: RootOperand> Pow for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for MultipleValuesWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Round for MultipleValuesWithIndexOperand<O> {
    fn round(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Round,
            });
    }
}

impl<O: RootOperand> Ceil for MultipleValuesWithIndexOperand<O> {
    fn ceil(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Ceil,
            });
    }
}

impl<O: RootOperand> Floor for MultipleValuesWithIndexOperand<O> {
    fn floor(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Floor,
            });
    }
}

impl<O: RootOperand> Abs for MultipleValuesWithIndexOperand<O> {
    fn abs(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl<O: RootOperand> Sqrt for MultipleValuesWithIndexOperand<O> {
    fn sqrt(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Sqrt,
            });
    }
}

impl<O: RootOperand> Trim for MultipleValuesWithIndexOperand<O> {
    fn trim(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl<O: RootOperand> TrimStart for MultipleValuesWithIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl<O: RootOperand> TrimEnd for MultipleValuesWithIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl<O: RootOperand> Lowercase for MultipleValuesWithIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl<O: RootOperand> Uppercase for MultipleValuesWithIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl<O: RootOperand> Slice for MultipleValuesWithIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleValuesWithIndexOperation::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for MultipleValuesWithIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsString);
    }
}

impl<O: RootOperand> IsInt for MultipleValuesWithIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsInt);
    }
}

impl<O: RootOperand> IsFloat for MultipleValuesWithIndexOperand<O> {
    fn is_float(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsFloat);
    }
}

impl<O: RootOperand> IsBool for MultipleValuesWithIndexOperand<O> {
    fn is_bool(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsBool);
    }
}

impl<O: RootOperand> IsDateTime for MultipleValuesWithIndexOperand<O> {
    fn is_datetime(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsDateTime);
    }
}

impl<O: RootOperand> IsDuration for MultipleValuesWithIndexOperand<O> {
    fn is_duration(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsDuration);
    }
}

impl<O: RootOperand> IsNull for MultipleValuesWithIndexOperand<O> {
    fn is_null(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsNull);
    }
}

impl<O: RootOperand> IsMax for MultipleValuesWithIndexOperand<O> {
    fn is_max(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsMax);
    }
}

impl<O: RootOperand> IsMin for MultipleValuesWithIndexOperand<O> {
    fn is_min(&mut self) {
        self.operations
            .push(MultipleValuesWithIndexOperation::IsMin);
    }
}

impl<O: RootOperand> EitherOr for MultipleValuesWithIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleValuesWithIndexOperation::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for MultipleValuesWithIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesWithIndexOperation::Exclude { operand });
    }
}

impl<O: RootOperand> MultipleValuesWithIndexOperand<O> {
    pub(crate) fn new(context: MultipleValuesWithIndexContext<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn push_merge_operation(&mut self, operand: Wrapper<Self>) {
        self.operations
            .push(MultipleValuesWithIndexOperation::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<MultipleValuesWithIndexOperand<O>> {
    pub(crate) fn new(context: MultipleValuesWithIndexContext<O>) -> Self {
        MultipleValuesWithIndexOperand::new(context).into()
    }

    pub(crate) fn push_merge_operation(&self, operand: Wrapper<MultipleValuesWithIndexOperand<O>>) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

pub type NodeMultipleValuesWithoutIndexOperand = MultipleValuesWithoutIndexOperand<NodeOperand>;
pub type EdgeMultipleValuesWithoutIndexOperand = MultipleValuesWithoutIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleValuesWithoutIndexOperand<O: RootOperand> {
    pub(crate) context: MultipleValuesWithoutIndexContext<O>,
    operations: Vec<MultipleValuesWithoutIndexOperation<O>>,
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
        let values: BoxedIterator<_> = Box::new(values);

        self.operations
            .iter()
            .try_fold(values, |value_tuples, operation| {
                operation.evaluate(medrecord, value_tuples)
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
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Min for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Min,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Mean for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn mean(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Mean,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Median for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn median(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Median,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Mode for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn mode(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Mode,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Std for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn std(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Std,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Var for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn var(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Var,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
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
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Sum for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Sum,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Random for MultipleValuesWithoutIndexOperand<O> {
    type ReturnOperand = SingleValueWithoutIndexOperand<O>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            SingleValueWithoutIndexContext::MultipleValuesWithoutIndexOperand(self.deep_clone()),
            SingleKindWithoutIndex::Random,
        );

        self.operations
            .push(MultipleValuesWithoutIndexOperation::ValueOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> GreaterThan for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Div for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Div,
            },
        );
    }
}

impl<O: RootOperand> Pow for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for MultipleValuesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Round for MultipleValuesWithoutIndexOperand<O> {
    fn round(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Round,
            },
        );
    }
}

impl<O: RootOperand> Ceil for MultipleValuesWithoutIndexOperand<O> {
    fn ceil(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Ceil,
            },
        );
    }
}

impl<O: RootOperand> Floor for MultipleValuesWithoutIndexOperand<O> {
    fn floor(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Floor,
            },
        );
    }
}

impl<O: RootOperand> Abs for MultipleValuesWithoutIndexOperand<O> {
    fn abs(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            },
        );
    }
}

impl<O: RootOperand> Sqrt for MultipleValuesWithoutIndexOperand<O> {
    fn sqrt(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Sqrt,
            },
        );
    }
}

impl<O: RootOperand> Trim for MultipleValuesWithoutIndexOperand<O> {
    fn trim(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            },
        );
    }
}

impl<O: RootOperand> TrimStart for MultipleValuesWithoutIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            },
        );
    }
}

impl<O: RootOperand> TrimEnd for MultipleValuesWithoutIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            },
        );
    }
}

impl<O: RootOperand> Lowercase for MultipleValuesWithoutIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            },
        );
    }
}

impl<O: RootOperand> Uppercase for MultipleValuesWithoutIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations.push(
            MultipleValuesWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            },
        );
    }
}

impl<O: RootOperand> Slice for MultipleValuesWithoutIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for MultipleValuesWithoutIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsString);
    }
}

impl<O: RootOperand> IsInt for MultipleValuesWithoutIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsInt);
    }
}

impl<O: RootOperand> IsFloat for MultipleValuesWithoutIndexOperand<O> {
    fn is_float(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsFloat);
    }
}

impl<O: RootOperand> IsBool for MultipleValuesWithoutIndexOperand<O> {
    fn is_bool(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsBool);
    }
}

impl<O: RootOperand> IsDateTime for MultipleValuesWithoutIndexOperand<O> {
    fn is_datetime(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsDateTime);
    }
}

impl<O: RootOperand> IsDuration for MultipleValuesWithoutIndexOperand<O> {
    fn is_duration(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsDuration);
    }
}

impl<O: RootOperand> IsNull for MultipleValuesWithoutIndexOperand<O> {
    fn is_null(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsNull);
    }
}

impl<O: RootOperand> IsMax for MultipleValuesWithoutIndexOperand<O> {
    fn is_max(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsMax);
    }
}

impl<O: RootOperand> IsMin for MultipleValuesWithoutIndexOperand<O> {
    fn is_min(&mut self) {
        self.operations
            .push(MultipleValuesWithoutIndexOperation::IsMin);
    }
}

impl<O: RootOperand> EitherOr for MultipleValuesWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<MultipleValuesWithoutIndexOperand<O>>::new(self.context.clone());
        let mut or_operand =
            Wrapper::<MultipleValuesWithoutIndexOperand<O>>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleValuesWithoutIndexOperation::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for MultipleValuesWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(MultipleValuesWithoutIndexOperation::Exclude { operand });
    }
}

impl<O: RootOperand> MultipleValuesWithoutIndexOperand<O> {
    pub(crate) fn new(context: MultipleValuesWithoutIndexContext<O>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl<O: RootOperand> Wrapper<MultipleValuesWithoutIndexOperand<O>> {
    pub(crate) fn new(context: MultipleValuesWithoutIndexContext<O>) -> Self {
        MultipleValuesWithoutIndexOperand::new(context).into()
    }
}

pub type NodeSingleValueWithIndexOperand = SingleValueWithIndexOperand<NodeOperand>;
pub type EdgeSingleValueWithIndexOperand = SingleValueWithIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleValueWithIndexOperand<O: RootOperand> {
    context: MultipleValuesWithIndexOperand<O>,
    pub(crate) kind: SingleKindWithIndex,
    operations: Vec<SingleValueWithIndexOperation<O>>,
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
            SingleKindWithIndex::Max => MultipleValuesWithIndexOperation::<O>::get_max(values)?,
            SingleKindWithIndex::Min => MultipleValuesWithIndexOperation::<O>::get_min(values)?,
            SingleKindWithIndex::Random => {
                MultipleValuesWithIndexOperation::<O>::get_random(values)
            }
        })
    }
}

impl<O: RootOperand> GreaterThan for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueWithIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueWithIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl<O: RootOperand> Sub for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl<O: RootOperand> Mul for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl<O: RootOperand> Div for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Div,
            });
    }
}

impl<O: RootOperand> Pow for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl<O: RootOperand> Mod for SingleValueWithIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(SingleValueWithIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl<O: RootOperand> Round for SingleValueWithIndexOperand<O> {
    fn round(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Round,
            });
    }
}

impl<O: RootOperand> Ceil for SingleValueWithIndexOperand<O> {
    fn ceil(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Ceil,
            });
    }
}

impl<O: RootOperand> Floor for SingleValueWithIndexOperand<O> {
    fn floor(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Floor,
            });
    }
}

impl<O: RootOperand> Abs for SingleValueWithIndexOperand<O> {
    fn abs(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl<O: RootOperand> Sqrt for SingleValueWithIndexOperand<O> {
    fn sqrt(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Sqrt,
            });
    }
}

impl<O: RootOperand> Trim for SingleValueWithIndexOperand<O> {
    fn trim(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl<O: RootOperand> TrimStart for SingleValueWithIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl<O: RootOperand> TrimEnd for SingleValueWithIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl<O: RootOperand> Lowercase for SingleValueWithIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl<O: RootOperand> Uppercase for SingleValueWithIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl<O: RootOperand> Slice for SingleValueWithIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleValueWithIndexOperation::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for SingleValueWithIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::IsString);
    }
}

impl<O: RootOperand> IsInt for SingleValueWithIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations.push(SingleValueWithIndexOperation::IsInt);
    }
}

impl<O: RootOperand> IsFloat for SingleValueWithIndexOperand<O> {
    fn is_float(&mut self) {
        self.operations.push(SingleValueWithIndexOperation::IsFloat);
    }
}

impl<O: RootOperand> IsBool for SingleValueWithIndexOperand<O> {
    fn is_bool(&mut self) {
        self.operations.push(SingleValueWithIndexOperation::IsBool);
    }
}

impl<O: RootOperand> IsDateTime for SingleValueWithIndexOperand<O> {
    fn is_datetime(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::IsDateTime);
    }
}

impl<O: RootOperand> IsDuration for SingleValueWithIndexOperand<O> {
    fn is_duration(&mut self) {
        self.operations
            .push(SingleValueWithIndexOperation::IsDuration);
    }
}

impl<O: RootOperand> IsNull for SingleValueWithIndexOperand<O> {
    fn is_null(&mut self) {
        self.operations.push(SingleValueWithIndexOperation::IsNull);
    }
}

impl<O: RootOperand> EitherOr for SingleValueWithIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleValueWithIndexOperation::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for SingleValueWithIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<SingleValueWithIndexOperand<O>>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleValueWithIndexOperation::Exclude { operand });
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

    pub(crate) fn push_merge_operation(
        &mut self,
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) {
        self.operations
            .push(SingleValueWithIndexOperation::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueWithIndexOperand<O>> {
    pub(crate) fn new(
        context: MultipleValuesWithIndexOperand<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        SingleValueWithIndexOperand::new(context, kind).into()
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
    operations: Vec<SingleValueWithoutIndexOperation<O>>,
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
                MultipleValuesWithoutIndexOperation::<O>::get_max(values)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesWithoutIndexOperation::<O>::get_min(values)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesWithoutIndexOperation::<O>::get_mean(values)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesWithoutIndexOperation::<O>::get_median(values)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesWithoutIndexOperation::<O>::get_mode(values)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesWithoutIndexOperation::<O>::get_std(values)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesWithoutIndexOperation::<O>::get_var(values)?
            }
            SingleKindWithoutIndex::Count => {
                Some(MultipleValuesWithoutIndexOperation::<O>::get_count(values))
            }
            SingleKindWithoutIndex::Sum => {
                MultipleValuesWithoutIndexOperation::<O>::get_sum(values)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesWithoutIndexOperation::<O>::get_random(values)
            }
        };

        self.evaluate_forward(medrecord, value)
    }
}

impl<O: RootOperand> GreaterThan for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::SingleValueComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleValuesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::MultipleValuesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Div for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Div,
            },
        );
    }
}

impl<O: RootOperand> Pow for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for SingleValueWithoutIndexOperand<O> {
    type ComparisonOperand = SingleValueComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleValueWithoutIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Round for SingleValueWithoutIndexOperand<O> {
    fn round(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Round,
            });
    }
}

impl<O: RootOperand> Ceil for SingleValueWithoutIndexOperand<O> {
    fn ceil(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Ceil,
            });
    }
}

impl<O: RootOperand> Floor for SingleValueWithoutIndexOperand<O> {
    fn floor(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Floor,
            });
    }
}

impl<O: RootOperand> Abs for SingleValueWithoutIndexOperand<O> {
    fn abs(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl<O: RootOperand> Sqrt for SingleValueWithoutIndexOperand<O> {
    fn sqrt(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Sqrt,
            });
    }
}

impl<O: RootOperand> Trim for SingleValueWithoutIndexOperand<O> {
    fn trim(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl<O: RootOperand> TrimStart for SingleValueWithoutIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl<O: RootOperand> TrimEnd for SingleValueWithoutIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl<O: RootOperand> Lowercase for SingleValueWithoutIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl<O: RootOperand> Uppercase for SingleValueWithoutIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl<O: RootOperand> Slice for SingleValueWithoutIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleValueWithoutIndexOperation::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for SingleValueWithoutIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsString);
    }
}

impl<O: RootOperand> IsInt for SingleValueWithoutIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsInt);
    }
}

impl<O: RootOperand> IsFloat for SingleValueWithoutIndexOperand<O> {
    fn is_float(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsFloat);
    }
}

impl<O: RootOperand> IsBool for SingleValueWithoutIndexOperand<O> {
    fn is_bool(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsBool);
    }
}

impl<O: RootOperand> IsDateTime for SingleValueWithoutIndexOperand<O> {
    fn is_datetime(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsDateTime);
    }
}

impl<O: RootOperand> IsDuration for SingleValueWithoutIndexOperand<O> {
    fn is_duration(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsDuration);
    }
}

impl<O: RootOperand> IsNull for SingleValueWithoutIndexOperand<O> {
    fn is_null(&mut self) {
        self.operations
            .push(SingleValueWithoutIndexOperation::IsNull);
    }
}

impl<O: RootOperand> EitherOr for SingleValueWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
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
            .push(SingleValueWithoutIndexOperation::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for SingleValueWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<SingleValueWithoutIndexOperand<O>>::new(
            self.context.clone(),
            self.kind.clone(),
        );

        query(&mut operand);

        self.operations
            .push(SingleValueWithoutIndexOperation::Exclude { operand });
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

    pub(crate) fn push_merge_operation(
        &mut self,
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) {
        self.operations
            .push(SingleValueWithoutIndexOperation::Merge { operand });
    }
}

impl<O: RootOperand> Wrapper<SingleValueWithoutIndexOperand<O>> {
    pub(crate) fn new(
        context: SingleValueWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
        SingleValueWithoutIndexOperand::new(context, kind).into()
    }

    pub(crate) fn push_merge_operation(
        &self,
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

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
            operand_traits::{
                Abs, Add, Contains, Count, EitherOr, EndsWith, EqualTo, Exclude, GreaterThan,
                GreaterThanOrEqualTo, IsIn, IsInt, IsMax, IsMin, IsNotIn, IsString, LessThan,
                LessThanOrEqualTo, Lowercase, Max, Min, Mod, Mul, NotEqualTo, Pow, Random, Slice,
                StartsWith, Sub, Sum, ToValues, Trim, TrimEnd, TrimStart, Uppercase,
            },
            values::{MultipleValuesWithIndexContext, MultipleValuesWithIndexOperand},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeOperand, MedRecordAttribute, NodeOperand, Wrapper,
    },
    MedRecord,
};
use medmodels_utils::aliases::MrHashSet;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum SingleAttributeComparisonOperand {
    NodeSingleAttributeWithIndexOperand(NodeSingleAttributeWithIndexOperand),
    NodeSingleAttributeWithoutIndexOperand(NodeSingleAttributeWithoutIndexOperand),
    EdgeSingleAttributeWithIndexOperand(EdgeSingleAttributeWithIndexOperand),
    EdgeSingleAttributeWithoutIndexOperand(EdgeSingleAttributeWithoutIndexOperand),
    Attribute(MedRecordAttribute),
}

impl DeepClone for SingleAttributeComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeSingleAttributeWithIndexOperand(operand) => {
                Self::NodeSingleAttributeWithIndexOperand(operand.deep_clone())
            }
            Self::NodeSingleAttributeWithoutIndexOperand(operand) => {
                Self::NodeSingleAttributeWithoutIndexOperand(operand.deep_clone())
            }
            Self::EdgeSingleAttributeWithIndexOperand(operand) => {
                Self::EdgeSingleAttributeWithIndexOperand(operand.deep_clone())
            }
            Self::EdgeSingleAttributeWithoutIndexOperand(operand) => {
                Self::EdgeSingleAttributeWithoutIndexOperand(operand.deep_clone())
            }
            Self::Attribute(attribute) => Self::Attribute(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeSingleAttributeWithIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<NodeSingleAttributeWithIndexOperand>) -> Self {
        Self::NodeSingleAttributeWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleAttributeWithIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<NodeSingleAttributeWithIndexOperand>) -> Self {
        Self::NodeSingleAttributeWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<NodeSingleAttributeWithoutIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<NodeSingleAttributeWithoutIndexOperand>) -> Self {
        Self::NodeSingleAttributeWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeSingleAttributeWithoutIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<NodeSingleAttributeWithoutIndexOperand>) -> Self {
        Self::NodeSingleAttributeWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleAttributeWithIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<EdgeSingleAttributeWithIndexOperand>) -> Self {
        Self::EdgeSingleAttributeWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleAttributeWithIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleAttributeWithIndexOperand>) -> Self {
        Self::EdgeSingleAttributeWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeSingleAttributeWithoutIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: Wrapper<EdgeSingleAttributeWithoutIndexOperand>) -> Self {
        Self::EdgeSingleAttributeWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeSingleAttributeWithoutIndexOperand>> for SingleAttributeComparisonOperand {
    fn from(value: &Wrapper<EdgeSingleAttributeWithoutIndexOperand>) -> Self {
        Self::EdgeSingleAttributeWithoutIndexOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeSingleAttributeWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::NodeSingleAttributeWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::EdgeSingleAttributeWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|attribute| attribute.1),
            Self::EdgeSingleAttributeWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?
            }
            Self::Attribute(attribute) => Some(attribute.clone()),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesComparisonOperand {
    NodeMultipleAttributesWithIndexOperand(NodeMultipleAttributesWithIndexOperand),
    NodeMultipleAttributesWithoutIndexOperand(NodeMultipleAttributesWithoutIndexOperand),
    EdgeMultipleAttributesWithIndexOperand(EdgeMultipleAttributesWithIndexOperand),
    EdgeMultipleAttributesWithoutIndexOperand(EdgeMultipleAttributesWithoutIndexOperand),
    Attributes(MrHashSet<MedRecordAttribute>),
}

impl DeepClone for MultipleAttributesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeMultipleAttributesWithIndexOperand(operand) => {
                Self::NodeMultipleAttributesWithIndexOperand(operand.deep_clone())
            }
            Self::NodeMultipleAttributesWithoutIndexOperand(operand) => {
                Self::NodeMultipleAttributesWithoutIndexOperand(operand.deep_clone())
            }
            Self::EdgeMultipleAttributesWithIndexOperand(operand) => {
                Self::EdgeMultipleAttributesWithIndexOperand(operand.deep_clone())
            }
            Self::EdgeMultipleAttributesWithoutIndexOperand(operand) => {
                Self::EdgeMultipleAttributesWithoutIndexOperand(operand.deep_clone())
            }
            Self::Attributes(attribute) => Self::Attributes(attribute.clone()),
        }
    }
}

impl From<Wrapper<NodeMultipleAttributesWithIndexOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<NodeMultipleAttributesWithIndexOperand>) -> Self {
        Self::NodeMultipleAttributesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleAttributesWithIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<NodeMultipleAttributesWithIndexOperand>) -> Self {
        Self::NodeMultipleAttributesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}
impl From<Wrapper<NodeMultipleAttributesWithoutIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: Wrapper<NodeMultipleAttributesWithoutIndexOperand>) -> Self {
        Self::NodeMultipleAttributesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeMultipleAttributesWithoutIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<NodeMultipleAttributesWithoutIndexOperand>) -> Self {
        Self::NodeMultipleAttributesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleAttributesWithIndexOperand>> for MultipleAttributesComparisonOperand {
    fn from(value: Wrapper<EdgeMultipleAttributesWithIndexOperand>) -> Self {
        Self::EdgeMultipleAttributesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleAttributesWithIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<EdgeMultipleAttributesWithIndexOperand>) -> Self {
        Self::EdgeMultipleAttributesWithIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<Wrapper<EdgeMultipleAttributesWithoutIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: Wrapper<EdgeMultipleAttributesWithoutIndexOperand>) -> Self {
        Self::EdgeMultipleAttributesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeMultipleAttributesWithoutIndexOperand>>
    for MultipleAttributesComparisonOperand
{
    fn from(value: &Wrapper<EdgeMultipleAttributesWithoutIndexOperand>) -> Self {
        Self::EdgeMultipleAttributesWithoutIndexOperand(value.0.read_or_panic().deep_clone())
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
            Self::NodeMultipleAttributesWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::NodeMultipleAttributesWithoutIndexOperand(operand) => {
                operand.evaluate_backward(medrecord)?.collect()
            }
            Self::EdgeMultipleAttributesWithIndexOperand(operand) => operand
                .evaluate_backward(medrecord)?
                .map(|(_, attribute)| attribute)
                .collect(),
            Self::EdgeMultipleAttributesWithoutIndexOperand(operand) => {
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

impl<O: RootOperand> Max for AttributesTreeOperand<O> {
    type ReturnOperand = MultipleAttributesWithIndexOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
            MultipleKind::Max,
        );

        self.operations
            .push(AttributesTreeOperation::AttributesOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Min for AttributesTreeOperand<O> {
    type ReturnOperand = MultipleAttributesWithIndexOperand<O>;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
            MultipleKind::Min,
        );

        self.operations
            .push(AttributesTreeOperation::AttributesOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for AttributesTreeOperand<O> {
    type ReturnOperand = MultipleAttributesWithIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
            MultipleKind::Count,
        );

        self.operations
            .push(AttributesTreeOperation::AttributesOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Sum for AttributesTreeOperand<O> {
    type ReturnOperand = MultipleAttributesWithIndexOperand<O>;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
            MultipleKind::Sum,
        );

        self.operations
            .push(AttributesTreeOperation::AttributesOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Random for AttributesTreeOperand<O> {
    type ReturnOperand = MultipleAttributesWithIndexOperand<O>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleAttributesWithIndexContext::AttributesTree(self.deep_clone()),
            MultipleKind::Random,
        );

        self.operations
            .push(AttributesTreeOperation::AttributesOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> GreaterThan for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            AttributesTreeOperation::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for AttributesTreeOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            AttributesTreeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for AttributesTreeOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            AttributesTreeOperation::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(AttributesTreeOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl<O: RootOperand> Sub for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(AttributesTreeOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl<O: RootOperand> Mul for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(AttributesTreeOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl<O: RootOperand> Pow for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(AttributesTreeOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl<O: RootOperand> Mod for AttributesTreeOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(AttributesTreeOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl<O: RootOperand> Abs for AttributesTreeOperand<O> {
    fn abs(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl<O: RootOperand> Trim for AttributesTreeOperand<O> {
    fn trim(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl<O: RootOperand> TrimStart for AttributesTreeOperand<O> {
    fn trim_start(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl<O: RootOperand> TrimEnd for AttributesTreeOperand<O> {
    fn trim_end(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl<O: RootOperand> Lowercase for AttributesTreeOperand<O> {
    fn lowercase(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl<O: RootOperand> Uppercase for AttributesTreeOperand<O> {
    fn uppercase(&mut self) {
        self.operations
            .push(AttributesTreeOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl<O: RootOperand> Slice for AttributesTreeOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(AttributesTreeOperation::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for AttributesTreeOperand<O> {
    fn is_string(&mut self) {
        self.operations.push(AttributesTreeOperation::IsString);
    }
}

impl<O: RootOperand> IsInt for AttributesTreeOperand<O> {
    fn is_int(&mut self) {
        self.operations.push(AttributesTreeOperation::IsInt);
    }
}

impl<O: RootOperand> IsMax for AttributesTreeOperand<O> {
    fn is_max(&mut self) {
        self.operations.push(AttributesTreeOperation::IsMax);
    }
}

impl<O: RootOperand> IsMin for AttributesTreeOperand<O> {
    fn is_min(&mut self) {
        self.operations.push(AttributesTreeOperation::IsMin);
    }
}

impl<O: RootOperand> EitherOr for AttributesTreeOperand<O> {
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

        self.operations.push(AttributesTreeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl<O: RootOperand> Exclude for AttributesTreeOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(AttributesTreeOperation::Exclude { operand });
    }
}

impl<O: RootOperand> AttributesTreeOperand<O> {
    pub(crate) fn new(context: O) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl<O: RootOperand> Wrapper<AttributesTreeOperand<O>> {
    pub(crate) fn new(context: O) -> Self {
        AttributesTreeOperand::new(context).into()
    }
}

pub type NodeMultipleAttributesWithIndexOperand = MultipleAttributesWithIndexOperand<NodeOperand>;
pub type EdgeMultipleAttributesWithIndexOperand = MultipleAttributesWithIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleAttributesWithIndexOperand<O: RootOperand> {
    context: MultipleAttributesWithIndexContext<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleAttributesWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for MultipleAttributesWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleAttributesWithIndexOperand<O> {
    type ReturnValue = BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.get_attributes(medrecord, &self.kind)?;

        self.evaluate_forward(medrecord, attributes)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for MultipleAttributesWithIndexOperand<O> {
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

impl<O: RootOperand> Max for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithIndexOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Max);

        self.operations
            .push(MultipleAttributesOperationWithIndex::AttributeOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Min for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithIndexOperand<O>;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Min);

        self.operations
            .push(MultipleAttributesOperationWithIndex::AttributeOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Count for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Count,
        );

        self.operations.push(
            MultipleAttributesOperationWithIndex::AttributeOperationWithoutIndex {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Sum for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Sum,
        );

        self.operations.push(
            MultipleAttributesOperationWithIndex::AttributeOperationWithoutIndex {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Random for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithIndexOperand<O>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKindWithIndex::Random);

        self.operations
            .push(MultipleAttributesOperationWithIndex::AttributeOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> GreaterThan for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Pow for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for MultipleAttributesWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Abs for MultipleAttributesWithIndexOperand<O> {
    fn abs(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            },
        );
    }
}

impl<O: RootOperand> Trim for MultipleAttributesWithIndexOperand<O> {
    fn trim(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            },
        );
    }
}

impl<O: RootOperand> TrimStart for MultipleAttributesWithIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            },
        );
    }
}

impl<O: RootOperand> TrimEnd for MultipleAttributesWithIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            },
        );
    }
}

impl<O: RootOperand> Lowercase for MultipleAttributesWithIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            },
        );
    }
}

impl<O: RootOperand> Uppercase for MultipleAttributesWithIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            },
        );
    }
}

impl<O: RootOperand> ToValues for MultipleAttributesWithIndexOperand<O> {
    type ReturnOperand = MultipleValuesWithIndexOperand<O>;

    fn to_values(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            MultipleValuesWithIndexContext::MultipleAttributesOperand(self.deep_clone()),
        );

        self.operations
            .push(MultipleAttributesOperationWithIndex::ToValues {
                operand: operand.clone(),
            });

        operand
    }
}

impl<O: RootOperand> Slice for MultipleAttributesWithIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for MultipleAttributesWithIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::IsString);
    }
}

impl<O: RootOperand> IsInt for MultipleAttributesWithIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::IsInt);
    }
}

impl<O: RootOperand> IsMax for MultipleAttributesWithIndexOperand<O> {
    fn is_max(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::IsMax);
    }
}

impl<O: RootOperand> IsMin for MultipleAttributesWithIndexOperand<O> {
    fn is_min(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithIndex::IsMin);
    }
}

impl<O: RootOperand> EitherOr for MultipleAttributesWithIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleAttributesOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for MultipleAttributesWithIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperationWithIndex::Exclude { operand });
    }
}

impl<O: RootOperand> MultipleAttributesWithIndexOperand<O> {
    pub(crate) fn new(context: MultipleAttributesWithIndexContext<O>, kind: MultipleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }
}

impl<O: RootOperand> Wrapper<MultipleAttributesWithIndexOperand<O>> {
    pub(crate) fn new(context: MultipleAttributesWithIndexContext<O>, kind: MultipleKind) -> Self {
        MultipleAttributesWithIndexOperand::new(context, kind).into()
    }
}

pub type NodeMultipleAttributesWithoutIndexOperand =
    MultipleAttributesWithoutIndexOperand<NodeOperand>;
pub type EdgeMultipleAttributesWithoutIndexOperand =
    MultipleAttributesWithoutIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct MultipleAttributesWithoutIndexOperand<O: RootOperand> {
    context: MultipleAttributesWithoutIndexContext<O>,
    pub(crate) kind: MultipleKind,
    operations: Vec<MultipleAttributesOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithoutIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for MultipleAttributesWithoutIndexOperand<O> {
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
    for MultipleAttributesWithoutIndexOperand<O>
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnValue = BoxedIterator<'a, MedRecordAttribute>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.get_attributes(medrecord)?;

        self.evaluate_forward(medrecord, attributes)
    }
}

impl<O: RootOperand> Max for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithoutIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Max,
        );

        self.operations.push(
            MultipleAttributesOperationWithoutIndex::AttributeOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Min for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithoutIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Min,
        );

        self.operations.push(
            MultipleAttributesOperationWithoutIndex::AttributeOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Count for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithoutIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Count,
        );

        self.operations.push(
            MultipleAttributesOperationWithoutIndex::AttributeOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Sum for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithoutIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Sum,
        );

        self.operations.push(
            MultipleAttributesOperationWithoutIndex::AttributeOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> Random for MultipleAttributesWithoutIndexOperand<O> {
    type ReturnOperand = SingleAttributeWithoutIndexOperand<O>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            SingleAttributeWithoutIndexContext::MultipleAttributesWithoutIndexOperand(
                self.deep_clone(),
            ),
            SingleKindWithoutIndex::Random,
        );

        self.operations.push(
            MultipleAttributesOperationWithoutIndex::AttributeOperation {
                operand: operand.clone(),
            },
        );

        operand
    }
}

impl<O: RootOperand> GreaterThan for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Pow for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for MultipleAttributesWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Abs for MultipleAttributesWithoutIndexOperand<O> {
    fn abs(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            },
        );
    }
}

impl<O: RootOperand> Trim for MultipleAttributesWithoutIndexOperand<O> {
    fn trim(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            },
        );
    }
}

impl<O: RootOperand> TrimStart for MultipleAttributesWithoutIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            },
        );
    }
}

impl<O: RootOperand> TrimEnd for MultipleAttributesWithoutIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            },
        );
    }
}

impl<O: RootOperand> Lowercase for MultipleAttributesWithoutIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            },
        );
    }
}

impl<O: RootOperand> Uppercase for MultipleAttributesWithoutIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations.push(
            MultipleAttributesOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            },
        );
    }
}

impl<O: RootOperand> Slice for MultipleAttributesWithoutIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for MultipleAttributesWithoutIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::IsString);
    }
}

impl<O: RootOperand> IsInt for MultipleAttributesWithoutIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::IsInt);
    }
}

impl<O: RootOperand> IsMax for MultipleAttributesWithoutIndexOperand<O> {
    fn is_max(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::IsMax);
    }
}

impl<O: RootOperand> IsMin for MultipleAttributesWithoutIndexOperand<O> {
    fn is_min(&mut self) {
        self.operations
            .push(MultipleAttributesOperationWithoutIndex::IsMin);
    }
}

impl<O: RootOperand> EitherOr for MultipleAttributesWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(MultipleAttributesOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for MultipleAttributesWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(MultipleAttributesOperationWithoutIndex::Exclude { operand });
    }
}

impl<O: RootOperand> MultipleAttributesWithoutIndexOperand<O> {
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
}

impl<O: RootOperand> Wrapper<MultipleAttributesWithoutIndexOperand<O>> {
    pub(crate) fn new(
        context: MultipleAttributesWithoutIndexContext<O>,
        kind: MultipleKind,
    ) -> Self {
        MultipleAttributesWithoutIndexOperand::new(context, kind).into()
    }
}

pub type NodeSingleAttributeWithIndexOperand = SingleAttributeWithIndexOperand<NodeOperand>;
pub type EdgeSingleAttributeWithIndexOperand = SingleAttributeWithIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleAttributeWithIndexOperand<O: RootOperand> {
    context: MultipleAttributesWithIndexOperand<O>,
    pub(crate) kind: SingleKindWithIndex,
    operations: Vec<SingleAttributeOperationWithIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleAttributeWithIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleAttributeWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleAttributeWithIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleAttributeWithIndexOperand<O> {
    type ReturnValue = <Self as EvaluateForward<'a>>::ReturnValue;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let attributes = self.context.evaluate_backward(medrecord)?;

        let attribute = self.reduce_input(attributes)?;

        self.evaluate_forward(medrecord, attribute)
    }
}

impl<'a, O: 'a + RootOperand> ReduceInput<'a> for SingleAttributeWithIndexOperand<O> {
    type Context = MultipleAttributesWithIndexOperand<O>;

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

impl<O: RootOperand> GreaterThan for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Pow for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for SingleAttributeWithIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Abs for SingleAttributeWithIndexOperand<O> {
    fn abs(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            },
        );
    }
}

impl<O: RootOperand> Trim for SingleAttributeWithIndexOperand<O> {
    fn trim(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            },
        );
    }
}

impl<O: RootOperand> TrimStart for SingleAttributeWithIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            },
        );
    }
}

impl<O: RootOperand> TrimEnd for SingleAttributeWithIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            },
        );
    }
}

impl<O: RootOperand> Lowercase for SingleAttributeWithIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            },
        );
    }
}

impl<O: RootOperand> Uppercase for SingleAttributeWithIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            },
        );
    }
}

impl<O: RootOperand> Slice for SingleAttributeWithIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleAttributeOperationWithIndex::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for SingleAttributeWithIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(SingleAttributeOperationWithIndex::IsString);
    }
}

impl<O: RootOperand> IsInt for SingleAttributeWithIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(SingleAttributeOperationWithIndex::IsInt);
    }
}

impl<O: RootOperand> EitherOr for SingleAttributeWithIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleAttributeOperationWithIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for SingleAttributeWithIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperationWithIndex::Exclude { operand });
    }
}

impl<O: RootOperand> SingleAttributeWithIndexOperand<O> {
    pub(crate) fn new(
        context: MultipleAttributesWithIndexOperand<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }
}

impl<O: RootOperand> Wrapper<SingleAttributeWithIndexOperand<O>> {
    pub(crate) fn new(
        context: MultipleAttributesWithIndexOperand<O>,
        kind: SingleKindWithIndex,
    ) -> Self {
        SingleAttributeWithIndexOperand::new(context, kind).into()
    }
}

pub type NodeSingleAttributeWithoutIndexOperand = SingleAttributeWithoutIndexOperand<NodeOperand>;
pub type EdgeSingleAttributeWithoutIndexOperand = SingleAttributeWithoutIndexOperand<EdgeOperand>;

#[derive(Debug, Clone)]
pub struct SingleAttributeWithoutIndexOperand<O: RootOperand> {
    context: SingleAttributeWithoutIndexContext<O>,
    pub(crate) kind: SingleKindWithoutIndex,
    operations: Vec<SingleAttributeOperationWithoutIndex<O>>,
}

impl<O: RootOperand> DeepClone for SingleAttributeWithoutIndexOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a, O: 'a + RootOperand> EvaluateForward<'a> for SingleAttributeWithoutIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateForwardGrouped<'a> for SingleAttributeWithoutIndexOperand<O> {
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

impl<'a, O: 'a + RootOperand> EvaluateBackward<'a> for SingleAttributeWithoutIndexOperand<O> {
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

impl<O: RootOperand> GreaterThan for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            },
        );
    }
}

impl<O: RootOperand> GreaterThanOrEqualTo for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> LessThan for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            },
        );
    }
}

impl<O: RootOperand> LessThanOrEqualTo for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            },
        );
    }
}

impl<O: RootOperand> EqualTo for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            },
        );
    }
}

impl<O: RootOperand> NotEqualTo for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            },
        );
    }
}

impl<O: RootOperand> StartsWith for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            },
        );
    }
}

impl<O: RootOperand> EndsWith for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            },
        );
    }
}

impl<O: RootOperand> Contains for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::SingleAttributeComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            },
        );
    }
}

impl<O: RootOperand> IsIn for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsIn,
            },
        );
    }
}

impl<O: RootOperand> IsNotIn for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = MultipleAttributesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, attributes: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::MultipleAttributesComparisonOperation {
                operand: attributes.into(),
                kind: MultipleComparisonKind::IsNotIn,
            },
        );
    }
}

impl<O: RootOperand> Add for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            },
        );
    }
}

impl<O: RootOperand> Sub for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            },
        );
    }
}

impl<O: RootOperand> Mul for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            },
        );
    }
}

impl<O: RootOperand> Pow for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            },
        );
    }
}

impl<O: RootOperand> Mod for SingleAttributeWithoutIndexOperand<O> {
    type ComparisonOperand = SingleAttributeComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            },
        );
    }
}

impl<O: RootOperand> Abs for SingleAttributeWithoutIndexOperand<O> {
    fn abs(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            },
        );
    }
}

impl<O: RootOperand> Trim for SingleAttributeWithoutIndexOperand<O> {
    fn trim(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            },
        );
    }
}

impl<O: RootOperand> TrimStart for SingleAttributeWithoutIndexOperand<O> {
    fn trim_start(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            },
        );
    }
}

impl<O: RootOperand> TrimEnd for SingleAttributeWithoutIndexOperand<O> {
    fn trim_end(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            },
        );
    }
}

impl<O: RootOperand> Lowercase for SingleAttributeWithoutIndexOperand<O> {
    fn lowercase(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            },
        );
    }
}

impl<O: RootOperand> Uppercase for SingleAttributeWithoutIndexOperand<O> {
    fn uppercase(&mut self) {
        self.operations.push(
            SingleAttributeOperationWithoutIndex::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            },
        );
    }
}

impl<O: RootOperand> Slice for SingleAttributeWithoutIndexOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(SingleAttributeOperationWithoutIndex::Slice(start..end));
    }
}

impl<O: RootOperand> IsString for SingleAttributeWithoutIndexOperand<O> {
    fn is_string(&mut self) {
        self.operations
            .push(SingleAttributeOperationWithoutIndex::IsString);
    }
}

impl<O: RootOperand> IsInt for SingleAttributeWithoutIndexOperand<O> {
    fn is_int(&mut self) {
        self.operations
            .push(SingleAttributeOperationWithoutIndex::IsInt);
    }
}

impl<O: RootOperand> EitherOr for SingleAttributeWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations
            .push(SingleAttributeOperationWithoutIndex::EitherOr {
                either: either_operand,
                or: or_operand,
            });
    }
}

impl<O: RootOperand> Exclude for SingleAttributeWithoutIndexOperand<O> {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(SingleAttributeOperationWithoutIndex::Exclude { operand });
    }
}

impl<O: RootOperand> SingleAttributeWithoutIndexOperand<O> {
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
}

impl<O: RootOperand> Wrapper<SingleAttributeWithoutIndexOperand<O>> {
    pub(crate) fn new(
        context: SingleAttributeWithoutIndexContext<O>,
        kind: SingleKindWithoutIndex,
    ) -> Self {
        SingleAttributeWithoutIndexOperand::new(context, kind).into()
    }
}

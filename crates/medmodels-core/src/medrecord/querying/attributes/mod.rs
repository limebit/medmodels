mod group_by;
mod operand;
mod operation;

use super::{
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
    EvaluateBackward,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::operation::AttributesTreeOperation, group_by::GroupOperand, BoxedIterator,
            DeepClone, RootOperand,
        },
        Attributes, EdgeIndex, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
pub use operand::{
    AttributesTreeOperand, EdgeAttributesTreeOperand, EdgeMultipleAttributesWithIndexOperand,
    EdgeMultipleAttributesWithoutIndexOperand, EdgeSingleAttributeWithIndexOperand,
    EdgeSingleAttributeWithoutIndexOperand, MultipleAttributesComparisonOperand,
    MultipleAttributesWithIndexOperand, MultipleAttributesWithoutIndexOperand,
    NodeAttributesTreeOperand, NodeMultipleAttributesWithIndexOperand,
    NodeMultipleAttributesWithoutIndexOperand, NodeSingleAttributeWithIndexOperand,
    NodeSingleAttributeWithoutIndexOperand, SingleAttributeComparisonOperand,
    SingleAttributeWithIndexOperand, SingleAttributeWithoutIndexOperand,
};
pub use operation::{
    MultipleAttributesWithIndexOperation, MultipleAttributesWithoutIndexOperation,
};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum AttributesTreeContext<O: RootOperand> {
    Operand(O),
    GroupByOperand(GroupOperand<AttributesTreeOperand<O>>),
}

impl<O: RootOperand> AttributesTreeContext<O> {
    pub(crate) fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        match self {
            Self::Operand(operand) => Ok(Box::new(operand.get_attributes(medrecord)?)),
            Self::GroupByOperand(operand) => Ok(Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .flat_map(|(_, attributes)| attributes),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesWithIndexContext<O: RootOperand> {
    AttributesTree {
        operand: AttributesTreeOperand<O>,
        kind: MultipleKind,
    },
    SingleAttributeWithIndexGroupByOperand(GroupOperand<SingleAttributeWithIndexOperand<O>>),
    MultipleAttributesWithIndexGroupByOperand(GroupOperand<MultipleAttributesWithIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::AttributesTree { operand, kind } => Self::AttributesTree {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::SingleAttributeWithIndexGroupByOperand(operand) => {
                Self::SingleAttributeWithIndexGroupByOperand(operand.deep_clone())
            }
            Self::MultipleAttributesWithIndexGroupByOperand(operand) => {
                Self::MultipleAttributesWithIndexGroupByOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> MultipleAttributesWithIndexContext<O> {
    pub(crate) fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let values: BoxedIterator<_> = match self {
            Self::AttributesTree { operand, kind } => {
                let attributes = operand.evaluate_backward(medrecord)?;

                match kind {
                    MultipleKind::Max => {
                        Box::new(AttributesTreeOperation::<O>::get_max(attributes)?)
                    }
                    MultipleKind::Min => {
                        Box::new(AttributesTreeOperation::<O>::get_min(attributes)?)
                    }
                    MultipleKind::Count => {
                        Box::new(AttributesTreeOperation::<O>::get_count(attributes)?)
                    }
                    MultipleKind::Sum => {
                        Box::new(AttributesTreeOperation::<O>::get_sum(attributes)?)
                    }
                    MultipleKind::Random => {
                        Box::new(AttributesTreeOperation::<O>::get_random(attributes)?)
                    }
                }
            }
            Self::SingleAttributeWithIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, attribute)| attribute),
            ),
            Self::MultipleAttributesWithIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .flat_map(|(_, attribute)| attribute),
            ),
        };

        Ok(values)
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesWithoutIndexContext<O: RootOperand> {
    GroupByOperand(GroupOperand<SingleAttributeWithoutIndexOperand<O>>),
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithoutIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::GroupByOperand(operand) => Self::GroupByOperand(operand.deep_clone()),
        }
    }
}

impl<O: RootOperand> MultipleAttributesWithoutIndexContext<O> {
    pub(crate) fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::GroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, attribute)| attribute),
            ),
        })
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeWithoutIndexContext<O: RootOperand> {
    MultipleAttributesWithIndexOperand(MultipleAttributesWithIndexOperand<O>),
    MultipleAttributesWithoutIndexOperand(MultipleAttributesWithoutIndexOperand<O>),
}

impl<O: RootOperand> DeepClone for SingleAttributeWithoutIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::MultipleAttributesWithIndexOperand(operand) => {
                Self::MultipleAttributesWithIndexOperand(operand.deep_clone())
            }
            Self::MultipleAttributesWithoutIndexOperand(operand) => {
                Self::MultipleAttributesWithoutIndexOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> SingleAttributeWithoutIndexContext<O> {
    pub(crate) fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::MultipleAttributesWithIndexOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .map(|(_, value)| value),
            ),
            Self::MultipleAttributesWithoutIndexOperand(operand) => {
                Box::new(operand.evaluate_backward(medrecord)?)
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum SingleKindWithIndex {
    Max,
    Min,
    Random,
}

#[derive(Debug, Clone)]
pub enum SingleKindWithoutIndex {
    Max,
    Min,
    Count,
    Sum,
    Random,
}

#[derive(Debug, Clone)]
pub enum MultipleKind {
    Max,
    Min,
    Count,
    Sum,
    Random,
}

#[derive(Debug, Clone)]
pub enum SingleComparisonKind {
    GreaterThan,
    GreaterThanOrEqualTo,
    LessThan,
    LessThanOrEqualTo,
    EqualTo,
    NotEqualTo,
    StartsWith,
    EndsWith,
    Contains,
}

#[derive(Debug, Clone)]
pub enum MultipleComparisonKind {
    IsIn,
    IsNotIn,
}

#[derive(Debug, Clone)]
pub enum BinaryArithmeticKind {
    Add,
    Sub,
    Mul,
    Pow,
    Mod,
}

impl Display for BinaryArithmeticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryArithmeticKind::Add => write!(f, "add"),
            BinaryArithmeticKind::Sub => write!(f, "sub"),
            BinaryArithmeticKind::Mul => write!(f, "mul"),
            BinaryArithmeticKind::Pow => write!(f, "pow"),
            BinaryArithmeticKind::Mod => write!(f, "mod"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UnaryArithmeticKind {
    Abs,
    Trim,
    TrimStart,
    TrimEnd,
    Lowercase,
    Uppercase,
}

pub trait GetAttributes {
    fn get_attributes<'a>(&'a self, medrecord: &'a MedRecord) -> MedRecordResult<&'a Attributes>;
}

impl GetAttributes for NodeIndex {
    fn get_attributes<'a>(&'a self, medrecord: &'a MedRecord) -> MedRecordResult<&'a Attributes> {
        medrecord.node_attributes(self)
    }
}

impl GetAttributes for EdgeIndex {
    fn get_attributes<'a>(&'a self, medrecord: &'a MedRecord) -> MedRecordResult<&'a Attributes> {
        medrecord.edge_attributes(self)
    }
}

impl<T: GetAttributes> GetAttributes for &T {
    fn get_attributes<'b>(&'b self, medrecord: &'b MedRecord) -> MedRecordResult<&'b Attributes> {
        (*self).get_attributes(medrecord)
    }
}

pub trait GetAllAttributes<I> {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a I, Vec<MedRecordAttribute>)> + 'a>
    where
        I: 'a;

    fn get_attributes_from_indices<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a I> + 'a,
    ) -> impl Iterator<Item = (&'a I, Vec<MedRecordAttribute>)> + 'a
    where
        I: 'a;
}

impl GetAllAttributes<NodeIndex> for NodeOperand {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)> + 'a>
    where
        NodeOperand: 'a,
    {
        let node_indices = self.evaluate_backward(medrecord)?;

        Ok(Self::get_attributes_from_indices(medrecord, node_indices))
    }

    fn get_attributes_from_indices<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> impl Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)> + 'a
    where
        NodeIndex: 'a,
    {
        NodeOperation::get_attributes(medrecord, indices)
    }
}

impl GetAllAttributes<EdgeIndex> for EdgeOperand {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a EdgeIndex, Vec<MedRecordAttribute>)> + 'a>
    where
        NodeOperand: 'a,
    {
        let edge_indices = self.evaluate_backward(medrecord)?;

        Ok(Self::get_attributes_from_indices(medrecord, edge_indices))
    }

    fn get_attributes_from_indices<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> impl Iterator<Item = (&'a EdgeIndex, Vec<MedRecordAttribute>)> + 'a
    where
        EdgeIndex: 'a,
    {
        EdgeOperation::get_attributes(medrecord, indices)
    }
}

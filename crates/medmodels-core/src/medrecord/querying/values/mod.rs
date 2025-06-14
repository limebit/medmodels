mod group_by;
mod operand;
mod operation;

use super::{
    attributes::{MultipleAttributesWithIndexOperand, MultipleAttributesWithIndexOperation},
    edges::EdgeOperand,
    group_by::GroupOperand,
    nodes::NodeOperand,
    BoxedIterator, EvaluateBackward, Index, RootOperand,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{querying::DeepClone, EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex},
    MedRecord,
};
pub use operand::{
    EdgeMultipleValuesWithIndexOperand, EdgeMultipleValuesWithoutIndexOperand,
    EdgeSingleValueWithIndexOperand, EdgeSingleValueWithoutIndexOperand,
    MultipleValuesComparisonOperand, MultipleValuesWithIndexOperand,
    MultipleValuesWithoutIndexOperand, NodeMultipleValuesWithIndexOperand,
    NodeMultipleValuesWithoutIndexOperand, NodeSingleValueWithIndexOperand,
    NodeSingleValueWithoutIndexOperand, SingleValueComparisonOperand, SingleValueWithIndexOperand,
    SingleValueWithoutIndexOperand,
};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum MultipleValuesWithIndexContext<O: RootOperand> {
    Operand((O, MedRecordAttribute)),
    MultipleAttributesOperand(MultipleAttributesWithIndexOperand<O>),
    SingleValueWithIndexGroupByOperand(GroupOperand<SingleValueWithIndexOperand<O>>),
    MultipleValuesWithIndexGroupByOperand(GroupOperand<MultipleValuesWithIndexOperand<O>>),
}

impl<O: RootOperand> MultipleValuesWithIndexContext<O> {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a>
    where
        O: 'a,
    {
        let values: BoxedIterator<_> = match self {
            Self::Operand((operand, attribute)) => {
                Box::new(operand.get_values(medrecord, attribute.clone())?)
            }
            Self::MultipleAttributesOperand(operand) => {
                let attributes = operand.evaluate_backward(medrecord)?;

                Box::new(MultipleAttributesWithIndexOperation::<O>::get_values(
                    medrecord, attributes,
                )?)
            }
            Self::SingleValueWithIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, attribute)| attribute),
            ),
            Self::MultipleValuesWithIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .flat_map(|(_, values)| values),
            ),
        };

        Ok(values)
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesWithoutIndexContext<O: RootOperand> {
    GroupByOperand(GroupOperand<SingleValueWithoutIndexOperand<O>>),
}

impl<O: RootOperand> MultipleValuesWithoutIndexContext<O> {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = MedRecordValue> + 'a>
    where
        O: 'a,
    {
        let values: BoxedIterator<_> = match self {
            Self::GroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, value)| value),
            ),
        };

        Ok(values)
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueWithoutIndexContext<O: RootOperand> {
    MultipleValuesWithIndexOperand(MultipleValuesWithIndexOperand<O>),
    MultipleValuesWithoutIndexOperand(MultipleValuesWithoutIndexOperand<O>),
}

impl<O: RootOperand> DeepClone for SingleValueWithoutIndexContext<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::MultipleValuesWithIndexOperand(operand) => {
                Self::MultipleValuesWithIndexOperand(operand.deep_clone())
            }
            Self::MultipleValuesWithoutIndexOperand(operand) => {
                Self::MultipleValuesWithoutIndexOperand(operand.deep_clone())
            }
        }
    }
}

impl<O: RootOperand> SingleValueWithoutIndexContext<O> {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::MultipleValuesWithIndexOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .map(|(_, value)| value),
            ),
            Self::MultipleValuesWithoutIndexOperand(operand) => {
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
    Mean,
    Median,
    Mode,
    Std,
    Var,
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
    Div,
    Pow,
    Mod,
}

impl Display for BinaryArithmeticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryArithmeticKind::Add => write!(f, "add"),
            BinaryArithmeticKind::Sub => write!(f, "sub"),
            BinaryArithmeticKind::Mul => write!(f, "mul"),
            BinaryArithmeticKind::Div => write!(f, "div"),
            BinaryArithmeticKind::Pow => write!(f, "pow"),
            BinaryArithmeticKind::Mod => write!(f, "mod"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UnaryArithmeticKind {
    Round,
    Ceil,
    Floor,
    Abs,
    Sqrt,
    Trim,
    TrimStart,
    TrimEnd,
    Lowercase,
    Uppercase,
}

pub trait GetValues<I: Index> {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a I, MedRecordValue)> + 'a>
    where
        I: 'a;

    fn get_values_from_indices<'a>(
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
        indices: impl Iterator<Item = &'a I> + 'a,
    ) -> impl Iterator<Item = (&'a I, MedRecordValue)> + 'a
    where
        I: 'a;
}

impl GetValues<NodeIndex> for NodeOperand {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a NodeIndex, MedRecordValue)> + 'a>
    where
        NodeIndex: 'a,
    {
        let node_indices = self.evaluate_backward(medrecord)?;

        Ok(Self::get_values_from_indices(
            medrecord,
            attribute,
            node_indices,
        ))
    }

    fn get_values_from_indices<'a>(
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> impl Iterator<Item = (&'a NodeIndex, MedRecordValue)> + 'a
    where
        NodeIndex: 'a,
    {
        node_indices.flat_map(move |node_index| {
            let attribute = medrecord
                .node_attributes(node_index)
                .expect("Node must exist")
                .get(&attribute)?
                .clone();

            Some((node_index, attribute))
        })
    }
}

impl GetValues<EdgeIndex> for EdgeOperand {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a EdgeIndex, MedRecordValue)> + 'a>
    where
        EdgeIndex: 'a,
    {
        let edge_indices = self.evaluate_backward(medrecord)?;

        Ok(Self::get_values_from_indices(
            medrecord,
            attribute,
            edge_indices,
        ))
    }

    fn get_values_from_indices<'a>(
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> impl Iterator<Item = (&'a EdgeIndex, MedRecordValue)> + 'a
    where
        EdgeIndex: 'a,
    {
        edge_indices.flat_map(move |edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        })
    }
}

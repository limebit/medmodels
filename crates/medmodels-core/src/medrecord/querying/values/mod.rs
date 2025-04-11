mod operand;
mod operation;

use super::{
    attributes::{GetAttributes, MultipleAttributesOperand, MultipleAttributesOperation},
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
    BoxedIterator, Index,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex},
    MedRecord,
};
pub use operand::{
    MultipleValuesComparisonOperand, MultipleValuesOperand, SingleValueComparisonOperand,
    SingleValueOperand,
};
pub use operation::MultipleValuesOperation;
use std::{fmt::Display, hash::Hash};

#[derive(Debug, Clone)]
pub enum SingleKind {
    Max,
    Min,
    Mean,
    Median,
    Mode,
    Std,
    Var,
    Count,
    Sum,
    First,
    Last,
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

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone)]
pub enum Context {
    NodeOperand(NodeOperand),
    EdgeOperand(EdgeOperand),
    MultipleAttributesOperand(MultipleAttributesOperand),
}

impl<'a> From<&'a NodeIndex> for Index<'a> {
    fn from(node_index: &'a NodeIndex) -> Self {
        Self::NodeIndex(node_index)
    }
}

impl<'a> From<&'a EdgeIndex> for Index<'a> {
    fn from(edge_index: &'a EdgeIndex) -> Self {
        Self::EdgeIndex(edge_index)
    }
}

impl Context {
    pub(crate) fn get_values<
        'a,
        T: 'a + Eq + Clone + Hash + GetAttributes + Display + Into<Index<'a>>,
    >(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<BoxedIterator<'a, (Index<'a>, MedRecordValue)>> {
        Ok(match self {
            Self::NodeOperand(node_operand) => {
                let node_indices = node_operand.evaluate(medrecord)?;

                Box::new(
                    NodeOperation::get_values(medrecord, node_indices, attribute)
                        .map(|(index, value)| (index.into(), value)),
                )
            }
            Self::EdgeOperand(edge_operand) => {
                let edge_indices = edge_operand.evaluate(medrecord)?;

                Box::new(
                    EdgeOperation::get_values(medrecord, edge_indices, attribute)
                        .map(|(index, value)| (index.into(), value)),
                )
            }
            Self::MultipleAttributesOperand(multiple_attributes_operand) => {
                let attributes = multiple_attributes_operand.evaluate::<T>(medrecord)?;

                Box::new(
                    MultipleAttributesOperation::get_values(medrecord, attributes)?
                        .map(|(index, value)| (index.into(), value)),
                )
            }
        })
    }
}

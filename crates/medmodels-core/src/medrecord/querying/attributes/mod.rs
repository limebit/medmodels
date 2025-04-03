mod operand;
mod operation;

use super::{
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
    BoxedIterator,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{Attributes, EdgeIndex, MedRecordAttribute, NodeIndex},
    MedRecord,
};
pub use operand::{
    AttributesTreeOperand, MultipleAttributesComparisonOperand, MultipleAttributesOperand,
    SingleAttributeComparisonOperand, SingleAttributeOperand,
};
pub use operation::{AttributesTreeOperation, MultipleAttributesOperation};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum SingleKind {
    Max,
    Min,
    Count,
    Sum,
    First,
    Last,
}

#[derive(Debug, Clone)]
pub enum MultipleKind {
    Max,
    Min,
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

pub(crate) trait GetAttributes {
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

#[derive(Debug, Clone)]
pub enum Context {
    NodeOperand(NodeOperand),
    EdgeOperand(EdgeOperand),
}

impl Context {
    pub(crate) fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, Vec<MedRecordAttribute>>> {
        Ok(match self {
            Self::NodeOperand(node_operand) => {
                let node_indices = node_operand.evaluate(medrecord, None)?;

                Box::new(
                    NodeOperation::get_attributes(medrecord, node_indices).map(|(_, value)| value),
                )
            }
            Self::EdgeOperand(edge_operand) => {
                let edge_indices = edge_operand.evaluate(medrecord, None)?;

                Box::new(
                    EdgeOperation::get_attributes(medrecord, edge_indices).map(|(_, value)| value),
                )
            }
        })
    }
}

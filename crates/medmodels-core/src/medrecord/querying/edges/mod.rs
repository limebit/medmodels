mod group_by;
mod operand;
mod operation;

use crate::{
    errors::MedRecordResult,
    medrecord::querying::{group_by::GroupOperand, BoxedIterator, EvaluateBackward},
    prelude::EdgeIndex,
    MedRecord,
};

use super::{
    nodes::{EdgeDirection, NodeOperand},
    DeepClone,
};
pub use group_by::EdgeOperandGroupDiscriminator;
pub use operand::{
    EdgeIndexComparisonOperand, EdgeIndexOperand, EdgeIndicesComparisonOperand, EdgeIndicesOperand,
    EdgeOperand,
};
pub use operation::EdgeOperation;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum EdgeOperandContext {
    Edges {
        operand: Box<NodeOperand>,
        kind: EdgeDirection,
    },
}

impl DeepClone for EdgeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeOperandContext::Edges { operand, kind } => EdgeOperandContext::Edges {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndicesOperandContext {
    EdgeOperand(EdgeOperand),
    GroupBy(GroupOperand<EdgeIndexOperand>),
}

impl DeepClone for EdgeIndicesOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeIndicesOperandContext::EdgeOperand(operand) => {
                EdgeIndicesOperandContext::EdgeOperand(operand.deep_clone())
            }
            EdgeIndicesOperandContext::GroupBy(group_by) => {
                EdgeIndicesOperandContext::GroupBy(group_by.deep_clone())
            }
        }
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndicesOperandContext {
    type ReturnValue = BoxedIterator<'a, EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        Ok(match self {
            EdgeIndicesOperandContext::EdgeOperand(operand) => {
                Box::new(operand.evaluate_backward(medrecord)?.cloned())
            }
            EdgeIndicesOperandContext::GroupBy(operand) => {
                Box::new(operand.evaluate_backward(medrecord)?.flatten())
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum SingleKind {
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

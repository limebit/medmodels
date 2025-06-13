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
    GroupBy {
        operand: Box<EdgeOperand>,
    },
}

impl DeepClone for EdgeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Edges { operand, kind } => Self::Edges {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::GroupBy { operand } => Self::GroupBy {
                operand: operand.deep_clone(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndicesOperandContext {
    EdgeOperand(EdgeOperand),
    EdgeIndexGroupByOperand(GroupOperand<EdgeIndexOperand>),
    EdgeIndicesGroupByOperand(GroupOperand<EdgeIndicesOperand>),
}

impl DeepClone for EdgeIndicesOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            Self::EdgeOperand(operand) => Self::EdgeOperand(operand.deep_clone()),
            Self::EdgeIndexGroupByOperand(operand) => {
                Self::EdgeIndexGroupByOperand(operand.deep_clone())
            }
            Self::EdgeIndicesGroupByOperand(operand) => {
                Self::EdgeIndicesGroupByOperand(operand.deep_clone())
            }
        }
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndicesOperandContext {
    type ReturnValue = BoxedIterator<'a, EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        Ok(match self {
            Self::EdgeOperand(operand) => Box::new(operand.evaluate_backward(medrecord)?.cloned()),
            Self::EdgeIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, index)| index),
            ),
            Self::EdgeIndicesGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .flat_map(|(_, index)| index),
            ),
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

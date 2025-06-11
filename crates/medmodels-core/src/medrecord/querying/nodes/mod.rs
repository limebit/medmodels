mod group_by;
mod operand;
mod operation;

use super::edges::EdgeOperand;
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{group_by::GroupOperand, BoxedIterator, DeepClone, EvaluateBackward},
    prelude::NodeIndex,
    MedRecord,
};
pub use group_by::NodeOperandGroupDiscriminator;
pub use operand::{
    NodeIndexComparisonOperand, NodeIndexOperand, NodeIndicesComparisonOperand, NodeIndicesOperand,
    NodeOperand,
};
pub use operation::{EdgeDirection, NodeOperation};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum NodeOperandContext {
    Neighbors {
        operand: Box<NodeOperand>,
        direction: EdgeDirection,
    },
    SourceNode {
        operand: EdgeOperand,
    },
    TargetNode {
        operand: EdgeOperand,
    },
    GroupBy {
        operand: Box<NodeOperand>,
    },
}

impl DeepClone for NodeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Neighbors { operand, direction } => Self::Neighbors {
                operand: operand.deep_clone(),
                direction: direction.clone(),
            },
            Self::SourceNode { operand } => Self::SourceNode {
                operand: operand.deep_clone(),
            },
            Self::TargetNode { operand } => Self::TargetNode {
                operand: operand.deep_clone(),
            },
            Self::GroupBy { operand } => Self::GroupBy {
                operand: operand.deep_clone(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndicesOperandContext {
    NodeOperand(NodeOperand),
    NodeIndexGroupByOperand(GroupOperand<NodeIndexOperand>),
    NodeIndicesGroupByOperand(GroupOperand<NodeIndicesOperand>),
}

impl DeepClone for NodeIndicesOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeOperand(operand) => Self::NodeOperand(operand.deep_clone()),
            Self::NodeIndexGroupByOperand(operand) => {
                Self::NodeIndexGroupByOperand(operand.deep_clone())
            }
            Self::NodeIndicesGroupByOperand(operand) => {
                Self::NodeIndicesGroupByOperand(operand.deep_clone())
            }
        }
    }
}

impl<'a> EvaluateBackward<'a> for NodeIndicesOperandContext {
    type ReturnValue = BoxedIterator<'a, NodeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        Ok(match self {
            Self::NodeOperand(operand) => Box::new(operand.evaluate_backward(medrecord)?.cloned()),
            Self::NodeIndexGroupByOperand(operand) => Box::new(
                operand
                    .evaluate_backward(medrecord)?
                    .filter_map(|(_, index)| index),
            ),
            Self::NodeIndicesGroupByOperand(operand) => Box::new(
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

#[derive(Debug, Clone)]
pub enum UnaryArithmeticKind {
    Abs,
    Trim,
    TrimStart,
    TrimEnd,
    Lowercase,
    Uppercase,
}

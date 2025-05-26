mod group_by;
mod operand;
mod operation;

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
pub enum Context {
    Edges {
        operand: Box<NodeOperand>,
        kind: EdgeDirection,
    },
}

impl DeepClone for Context {
    fn deep_clone(&self) -> Self {
        match self {
            Context::Edges { operand, kind } => Context::Edges {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
        }
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

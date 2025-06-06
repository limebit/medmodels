mod group_by;
mod operand;
mod operation;

use super::edges::EdgeOperand;
pub use operand::{
    NodeIndexComparisonOperand, NodeIndexOperand, NodeIndicesComparisonOperand, NodeIndicesOperand,
    NodeOperand,
};
pub use operation::{EdgeDirection, NodeOperation};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Context {
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

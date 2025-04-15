mod operand;
mod operation;

use super::{
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
};
use crate::{
    errors::MedRecordResult,
    medrecord::{EdgeIndex, MedRecordAttribute, NodeIndex},
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

pub(crate) trait GetAttributes<I> {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a I, Vec<MedRecordAttribute>)>>
    where
        I: 'a;
}

impl GetAttributes<NodeIndex> for NodeOperand {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)>>
    where
        NodeIndex: 'a,
    {
        Ok(NodeOperation::get_attributes(
            medrecord,
            self.evaluate(medrecord)?,
        ))
    }
}

impl GetAttributes<EdgeIndex> for EdgeOperand {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a EdgeIndex, Vec<MedRecordAttribute>)>>
    where
        EdgeIndex: 'a,
    {
        Ok(EdgeOperation::get_attributes(
            medrecord,
            self.evaluate(medrecord)?,
        ))
    }
}

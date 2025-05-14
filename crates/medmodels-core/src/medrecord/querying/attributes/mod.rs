mod operand;
mod operation;

use super::{
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
    EvaluateBackward,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{Attributes, EdgeIndex, MedRecordAttribute, NodeIndex},
    MedRecord,
};
pub use operand::{
    AttributesTreeOperand, EdgeAttributesTreeOperand, EdgeMultipleAttributesOperand,
    EdgeSingleAttributeOperand, MultipleAttributesComparisonOperand, MultipleAttributesOperand,
    NodeAttributesTreeOperand, NodeMultipleAttributesOperand, NodeSingleAttributeOperand,
    SingleAttributeComparisonOperand, SingleAttributeOperand,
};
pub use operation::MultipleAttributesOperation;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum SingleKind {
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
}

impl GetAllAttributes<NodeIndex> for NodeOperand {
    fn get_attributes<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)> + 'a>
    where
        NodeOperand: 'a,
    {
        Ok(NodeOperation::get_attributes(
            medrecord,
            self.evaluate_backward(medrecord)?,
        ))
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
        Ok(EdgeOperation::get_attributes(
            medrecord,
            self.evaluate_backward(medrecord)?,
        ))
    }
}

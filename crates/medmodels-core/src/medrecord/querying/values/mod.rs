mod operand;
mod operation;

use super::{
    attributes::{
        self, AttributesTreeOperation, MultipleAttributesOperand, MultipleAttributesOperation,
    },
    edges::{EdgeOperand, EdgeOperation},
    nodes::{NodeOperand, NodeOperation},
    BoxedIterator,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{MedRecordAttribute, MedRecordValue},
    MedRecord,
};
pub use operand::{
    MultipleValuesComparisonOperand, MultipleValuesOperand, SingleValueComparisonOperand,
    SingleValueOperand,
};
use std::fmt::Display;

macro_rules! get_attributes {
    ($operand:ident, $medrecord:ident, $operation:ident, $multiple_attributes_operand:ident) => {{
        let indices = $operand.evaluate($medrecord)?;

        let attributes = $operation::get_attributes($medrecord, indices);

        let attributes = $multiple_attributes_operand
            .context
            .evaluate($medrecord, attributes)?;

        let attributes: Box<dyn Iterator<Item = (_, MedRecordAttribute)>> =
            match $multiple_attributes_operand.kind {
                attributes::MultipleKind::Max => {
                    Box::new(AttributesTreeOperation::get_max(attributes)?)
                }
                attributes::MultipleKind::Min => {
                    Box::new(AttributesTreeOperation::get_min(attributes)?)
                }
                attributes::MultipleKind::Count => {
                    Box::new(AttributesTreeOperation::get_count(attributes)?)
                }
                attributes::MultipleKind::Sum => {
                    Box::new(AttributesTreeOperation::get_sum(attributes)?)
                }
                attributes::MultipleKind::First => {
                    Box::new(AttributesTreeOperation::get_first(attributes)?)
                }
                attributes::MultipleKind::Last => {
                    Box::new(AttributesTreeOperation::get_last(attributes)?)
                }
            };

        let attributes = $multiple_attributes_operand.evaluate($medrecord, attributes)?;

        Box::new(
            MultipleAttributesOperation::get_values($medrecord, attributes)?
                .map(|(_, value)| value),
        )
    }};
}

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

impl Context {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>> {
        Ok(match self {
            Self::NodeOperand(node_operand) => {
                let node_indices = node_operand.evaluate(medrecord)?;

                Box::new(
                    NodeOperation::get_values(medrecord, node_indices, attribute)
                        .map(|(_, value)| value),
                )
            }
            Self::EdgeOperand(edge_operand) => {
                let edge_indices = edge_operand.evaluate(medrecord)?;

                Box::new(
                    EdgeOperation::get_values(medrecord, edge_indices, attribute)
                        .map(|(_, value)| value),
                )
            }
            Self::MultipleAttributesOperand(multiple_attributes_operand) => {
                match &multiple_attributes_operand.context.context {
                    attributes::Context::NodeOperand(node_operand) => {
                        get_attributes!(
                            node_operand,
                            medrecord,
                            NodeOperation,
                            multiple_attributes_operand
                        )
                    }
                    attributes::Context::EdgeOperand(edge_operand) => {
                        get_attributes!(
                            edge_operand,
                            medrecord,
                            EdgeOperation,
                            multiple_attributes_operand
                        )
                    }
                }
            }
        })
    }
}

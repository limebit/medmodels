mod operand;
mod operation;

use super::{
    attributes::{MultipleAttributesOperand, MultipleAttributesOperation},
    edges::EdgeOperand,
    nodes::NodeOperand,
    BoxedIterator, Index, Operand,
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
use std::fmt::Display;

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

pub(crate) trait GetValues<I: Index> {
    fn get_values(
        &self,
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (I, MedRecordValue)>>;
}

impl GetValues<NodeIndex> for NodeOperand {
    fn get_values(
        &self,
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (NodeIndex, MedRecordValue)>> {
        let node_indices = self.evaluate(medrecord)?;

        Ok(node_indices.flat_map(move |node_index| {
            let attribute = medrecord
                .node_attributes(&node_index)
                .expect("Node must exist")
                .get(&attribute)?
                .clone();

            Some((node_index, attribute))
        }))
    }
}

impl GetValues<EdgeIndex> for EdgeOperand {
    fn get_values(
        &self,
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (EdgeIndex, MedRecordValue)>> {
        let edge_indices = self.evaluate(medrecord)?;

        Ok(edge_indices.flat_map(move |edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(&edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        }))
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone)]
pub enum Context<O: Operand> {
    Operand((O, MedRecordAttribute)),
    MultipleAttributesOperand(MultipleAttributesOperand<O>),
}

impl<O: Operand> Context<O> {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = (O::Index, MedRecordValue)>> {
        let values: BoxedIterator<(O::Index, MedRecordValue)> = match self {
            Self::Operand((operand, attribute)) => {
                Box::new(operand.get_values(medrecord, attribute.clone())?)
            }
            Self::MultipleAttributesOperand(multiple_attributes_operand) => {
                let attributes = multiple_attributes_operand.evaluate(medrecord)?;

                Box::new(
                    MultipleAttributesOperation::get_values(medrecord, attributes)?
                        .map(|(index, value)| (index.into(), value)),
                )
            }
        };

        Ok(values)
    }
}

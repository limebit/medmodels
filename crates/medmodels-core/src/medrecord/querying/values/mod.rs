mod operand;
mod operation;

use super::{
    attributes::MultipleAttributesOperand, edges::EdgeOperand, nodes::NodeOperand, BoxedIterator,
    Index, Operand,
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
pub use operation::MultipleValuesOperation;
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
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a I, MedRecordValue)>>
    where
        I: 'a;
}

impl GetValues<NodeIndex> for NodeOperand {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a NodeIndex, MedRecordValue)>>
    where
        NodeIndex: 'a,
    {
        let node_indices = self.evaluate(medrecord)?;

        Ok(node_indices.flat_map(move |node_index| {
            Some((
                node_index,
                medrecord
                    .node_attributes(node_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        }))
    }
}

impl GetValues<EdgeIndex> for EdgeOperand {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a EdgeIndex, MedRecordValue)>>
    where
        EdgeIndex: 'a,
    {
        let edge_indices = self.evaluate(medrecord)?;

        Ok(edge_indices.flat_map(move |edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        }))
    }
}

impl<O: Operand> GetValues<O::Index> for MultipleAttributesOperand<O> {
    fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordValue)>>
    where
        O::Index: 'a,
    {
        let attributes = self.evaluate(medrecord)?;

        // Ok(attributes
        //     .map(|(index, attribute)| {
        //         let value = index.get_attributes(medrecord)?.get(&attribute).ok_or(
        //             MedRecordError::QueryError(format!(
        //                 "Cannot find attribute {} for index {}",
        //                 attribute, index
        //             )),
        //         )?;

        //         Ok((index.clone(), value.clone()))
        //     })
        //     .collect::<MedRecordResult<Vec<_>>>()?
        //     .into_iter())

        todo!();

        Ok(std::iter::empty())
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone)]
pub enum Context<O: Operand> {
    Operand(O),
    MultipleAttributesOperand(MultipleAttributesOperand<O>),
}

impl<O: Operand> Context<O> {
    pub(crate) fn get_values<'a>(
        &'a self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> MedRecordResult<BoxedIterator<(&'a O::Index, MedRecordValue)>> {
        Ok(match self {
            Self::Operand(operand) => Box::new(operand.get_values(medrecord, attribute)?),
            Self::MultipleAttributesOperand(multiple_attributes_operand) => {
                Box::new(multiple_attributes_operand.get_values(medrecord, attribute)?)
            }
        })
    }
}

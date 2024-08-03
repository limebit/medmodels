#![allow(dead_code)]

use super::EdgeOperand;
use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperandContext, EvaluateOperation},
            values::{ComparisonOperand, ValuesOperand},
            wrapper::{OperandContext, Wrapper},
        },
        EdgeIndex, MedRecordAttribute,
    },
    MedRecord,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub enum EdgeValuesOperation {
    Max { operand: Wrapper<EdgeValueOperand> },
}

impl EvaluateOperation for EdgeValuesOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        _indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::Max { operand } => Box::new(Self::evaluate_max(medrecord, operand.clone())),
        }
    }
}

impl EdgeValuesOperation {
    fn evaluate_max(
        medrecord: &MedRecord,
        operand: Wrapper<EdgeValueOperand>,
    ) -> impl Iterator<Item = &EdgeIndex> {
        operand.evaluate(medrecord, None)
    }
}

#[derive(Debug, Clone)]
pub struct EdgeValuesOperand {
    context: OperandContext<EdgeOperand>,
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<EdgeValuesOperation>,
}

impl EvaluateOperand for EdgeValuesOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices = self.context.evaluate(medrecord);

        self.operations[0..end_index.unwrap_or(self.operations.len())]
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl EdgeValuesOperand {
    pub(crate) fn new(context: OperandContext<EdgeOperand>, attribute: MedRecordAttribute) -> Self {
        Self {
            context,
            attribute,
            operations: Vec::new(),
        }
    }

    pub fn max(&mut self, self_wrapper: &Wrapper<EdgeValuesOperand>) -> Wrapper<EdgeValueOperand> {
        let mut operand = EdgeValueOperand::new(self.attribute.clone());

        let context = EdgeValueOperation::MaxContext {
            context: OperandContext::new(self_wrapper.clone(), self.operations.len()),
            attribute: self.attribute.clone(),
        };

        operand.operations.push(context);

        let operand = Wrapper::from(operand);

        self.operations.push(EdgeValuesOperation::Max {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeValuesOperand> {
    pub(crate) fn new(context: OperandContext<EdgeOperand>, attribute: MedRecordAttribute) -> Self {
        Self(Rc::new(RefCell::new(EdgeValuesOperand::new(
            context, attribute,
        ))))
    }

    pub fn max(&self) -> Wrapper<EdgeValueOperand> {
        self.0.borrow_mut().max(self)
    }
}

#[derive(Debug, Clone)]
pub enum EdgeValueOperation {
    // If this operation is used, it is always the first operation of an operand.
    MaxContext {
        context: OperandContext<EdgeValuesOperand>,
        attribute: MedRecordAttribute,
    },

    LessThan {
        operand: ComparisonOperand,
        attribute: MedRecordAttribute,
    },
}

impl EvaluateOperation for EdgeValueOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::MaxContext { context, attribute } => Box::new(Self::evaluate_max_context(
                medrecord,
                context.clone(),
                attribute.clone(),
            )),
            Self::LessThan { operand, attribute } => Box::new(Self::evaluate_less_than(
                medrecord,
                indices,
                operand.clone(),
                attribute.clone(),
            )),
        }
    }
}

impl EdgeValueOperation {
    fn evaluate_max_context(
        medrecord: &MedRecord,
        context: OperandContext<EdgeValuesOperand>,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let edge_indices = context.evaluate(medrecord);

        let mut edge_attributes = edge_indices.filter_map(|edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?,
            ))
        });

        let Some(max) = edge_attributes.next() else {
            return Vec::new().into_iter();
        };

        let max_edge =
            edge_attributes.fold(max, |max, edge| if edge.1 > max.1 { edge } else { max });

        vec![max_edge.0].into_iter()
    }

    fn evaluate_less_than<'a>(
        medrecord: &'a MedRecord,
        mut edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: ComparisonOperand,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let Some(edge_index) = edge_indices.next() else {
            return Vec::new().into_iter();
        };
        let value = medrecord
            .edge_attributes(edge_index)
            .expect("Edge must exist")
            .get(&attribute)
            .expect("Attribute must exist");

        let ComparisonOperand::Multiple(comparison) = operand else {
            todo!()
        };

        let ValuesOperand::Edges(operand) = comparison else {
            todo!()
        };

        let comparison_edge_indices = operand.evaluate(medrecord, None);
        let comparison_attribute = operand.0.borrow().attribute.clone();

        let comparison_values = comparison_edge_indices.filter_map(|edge_index| {
            medrecord
                .edge_attributes(edge_index)
                .expect("Edge must exist")
                .get(&comparison_attribute)
        });

        for comparison_value in comparison_values {
            if value >= comparison_value {
                return Vec::new().into_iter();
            }
        }

        vec![edge_index].into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct EdgeValueOperand {
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<EdgeValueOperation>,
}

impl EvaluateOperand for EdgeValueOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices = medrecord.edge_indices();

        self.operations[0..end_index.unwrap_or(self.operations.len())]
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl EdgeValueOperand {
    pub fn new(attribute: MedRecordAttribute) -> Self {
        Self {
            attribute,
            operations: Vec::new(),
        }
    }

    pub fn less_than(&mut self, comparison: ComparisonOperand) {
        self.operations.push(EdgeValueOperation::LessThan {
            operand: comparison,
            attribute: self.attribute.clone(),
        });
    }
}

impl Wrapper<EdgeValueOperand> {
    pub(crate) fn new(attribute: MedRecordAttribute) -> Self {
        Self(Rc::new(RefCell::new(EdgeValueOperand::new(attribute))))
    }

    pub fn less_than<O>(&self, comparison: O)
    where
        O: Into<ComparisonOperand>,
    {
        self.0.borrow_mut().less_than(comparison.into());
    }
}

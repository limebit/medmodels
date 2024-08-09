#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::operation::{EdgeOperation, EdgeValueOperation, EdgeValuesOperation};
use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperandContext, EvaluateOperation},
            nodes::NodeOperand,
            values::ComparisonOperand,
            wrapper::{OperandContext, Wrapper},
        },
        EdgeIndex, MedRecordAttribute,
    },
    MedRecord,
};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    pub(crate) operations: Vec<EdgeOperation>,
}

impl EvaluateOperand for EdgeOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices =
            Box::new(medrecord.edge_indices()) as Box<dyn Iterator<Item = &'a EdgeIndex>>;

        self.operations[0..end_index.unwrap_or(self.operations.len())]
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl EdgeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn connects_to<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut node_operand = Wrapper::<NodeOperand>::new();

        query(&mut node_operand);

        self.operations.push(EdgeOperation::ConnectsTo {
            operand: node_operand,
        });
    }

    pub fn attribute(
        &mut self,
        attribute: MedRecordAttribute,
        self_wrapper: &Wrapper<EdgeOperand>,
    ) -> Wrapper<EdgeValuesOperand> {
        let operand = Wrapper::<EdgeValuesOperand>::new(
            OperandContext::new(self_wrapper.clone(), self.operations.len()),
            attribute,
        );

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(EdgeOperand::new())))
    }

    pub fn connects_to<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        self.0.borrow_mut().connects_to(query);
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<EdgeValuesOperand>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.borrow_mut().attribute(attribute.into(), self)
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

use super::{operation::EdgeValuesOperation, EdgeOperation};
use crate::{
    medrecord::{querying::nodes::NodeOperandWrapper, EdgeIndex, MedRecordAttribute},
    MedRecord,
};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    pub(crate) operations: Vec<EdgeOperation>,
}

impl EdgeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let edge_indices =
            Box::new(medrecord.edge_indices()) as Box<dyn Iterator<Item = &'a EdgeIndex>>;

        self.operations
            .iter()
            .fold(edge_indices, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }

    pub fn connects_to<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut NodeOperandWrapper),
    {
        let mut node_operand = NodeOperandWrapper::new();

        query(&mut node_operand);

        self.operations.push(EdgeOperation::ConnectsTo {
            operand: node_operand,
        });
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> EdgeValuesOperandWrapper {
        let operand = EdgeValuesOperandWrapper::new(self.clone().into(), attribute);

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EdgeOperandWrapper(pub(crate) Rc<RefCell<EdgeOperand>>);

impl From<EdgeOperand> for EdgeOperandWrapper {
    fn from(value: EdgeOperand) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}

impl EdgeOperandWrapper {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(EdgeOperand::new())))
    }

    pub fn connects_to<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut NodeOperandWrapper),
    {
        self.0.borrow_mut().connects_to(query);
    }

    pub fn attribute<A>(&self, attribute: A) -> EdgeValuesOperandWrapper
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.borrow_mut().attribute(attribute.into())
    }
}

#[derive(Debug, Clone)]
pub struct EdgeValuesOperand {
    operand: EdgeOperandWrapper,
    attribute: MedRecordAttribute,
    pub(crate) operations: Vec<EdgeValuesOperation>,
}

impl EdgeValuesOperand {
    pub(crate) fn new(operand: EdgeOperandWrapper, attribute: MedRecordAttribute) -> Self {
        Self {
            operand,
            attribute,
            operations: Vec::new(),
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EdgeValuesOperandWrapper(pub(crate) Rc<RefCell<EdgeValuesOperand>>);

impl From<EdgeValuesOperand> for EdgeValuesOperandWrapper {
    fn from(value: EdgeValuesOperand) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}

impl EdgeValuesOperandWrapper {
    pub(crate) fn new(operand: EdgeOperandWrapper, attribute: MedRecordAttribute) -> Self {
        Self(Rc::new(RefCell::new(EdgeValuesOperand::new(
            operand, attribute,
        ))))
    }
}

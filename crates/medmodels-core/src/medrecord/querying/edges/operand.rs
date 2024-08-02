use super::EdgeOperation;
use crate::{
    medrecord::{
        querying::{nodes::NodeOperandWrapper, values::ValuesOperandWrapper},
        EdgeIndex, MedRecordAttribute,
    },
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

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> ValuesOperandWrapper {
        let operand =
            ValuesOperandWrapper::new(EdgeOperandWrapper::from(self.clone()).into(), attribute);

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EdgeOperandWrapper(Rc<RefCell<EdgeOperand>>);

impl From<EdgeOperand> for EdgeOperandWrapper {
    fn from(value: EdgeOperand) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}

impl EdgeOperandWrapper {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(EdgeOperand::new())))
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        self.0.borrow().evaluate(medrecord)
    }

    pub fn connects_to<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut NodeOperandWrapper),
    {
        self.0.borrow_mut().connects_to(query);
    }

    pub fn attribute<A>(&self, attribute: A) -> ValuesOperandWrapper
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.borrow_mut().attribute(attribute.into())
    }
}

#![allow(dead_code)]

use super::{edges::EdgeOperandWrapper, nodes::NodeOperandWrapper};
use crate::medrecord::MedRecordAttribute;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub enum EdgeValuesOperation {}

#[derive(Debug, Clone)]
pub enum ValuesContext {
    Node(NodeOperandWrapper),
    Edge(EdgeOperandWrapper),
}

impl From<NodeOperandWrapper> for ValuesContext {
    fn from(operand: NodeOperandWrapper) -> Self {
        Self::Node(operand)
    }
}

impl From<EdgeOperandWrapper> for ValuesContext {
    fn from(operand: EdgeOperandWrapper) -> Self {
        Self::Edge(operand)
    }
}

#[derive(Debug, Clone)]
pub struct ValuesOperand {
    context: ValuesContext,
    attribute: MedRecordAttribute,
    operations: Vec<EdgeValuesOperation>,
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ValuesOperandWrapper(Rc<RefCell<ValuesOperand>>);

impl From<ValuesOperand> for ValuesOperandWrapper {
    fn from(operand: ValuesOperand) -> Self {
        Self(Rc::new(RefCell::new(operand)))
    }
}

impl ValuesOperandWrapper {
    pub fn new(context: ValuesContext, attribute: MedRecordAttribute) -> Self {
        Self(Rc::new(RefCell::new(ValuesOperand {
            context,
            attribute,
            operations: Vec::new(),
        })))
    }
}

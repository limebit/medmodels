#![allow(dead_code)]

use super::NodeOperand;
use crate::{
    medrecord::{
        querying::{evaluate::EvaluateOperand, values::ComparisonOperand, wrapper::Wrapper},
        MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub enum NodeValuesOperation {
    Max { operand: Wrapper<NodeValueOperand> },
}

#[derive(Debug, Clone)]
pub struct NodeValuesOperand {
    context: Wrapper<NodeOperand>,
    attribute: MedRecordAttribute,
    operations: Vec<NodeValuesOperation>,
}

impl EvaluateOperand for NodeValuesOperand {
    type Index = NodeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        self.context.evaluate(medrecord, end_index)
        // TODO
    }
}

impl NodeValuesOperand {
    pub(crate) fn new(context: Wrapper<NodeOperand>, attribute: MedRecordAttribute) -> Self {
        Self {
            context,
            attribute,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<NodeValuesOperand> {
    pub(crate) fn new(context: Wrapper<NodeOperand>, attribute: MedRecordAttribute) -> Self {
        Self(Rc::new(RefCell::new(NodeValuesOperand::new(
            context, attribute,
        ))))
    }
}

#[derive(Debug, Clone)]
pub enum NodeValueOperation {
    LessThan { operand: ComparisonOperand },
}

#[derive(Debug, Clone)]
pub struct NodeValueOperand {
    context: Wrapper<NodeValuesOperand>,
    operations: Vec<NodeValueOperation>,
}

impl NodeValueOperand {
    pub(crate) fn new(context: Wrapper<NodeValuesOperand>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<NodeValueOperand> {
    pub(crate) fn new(context: Wrapper<NodeValuesOperand>) -> Self {
        Self(Rc::new(RefCell::new(NodeValueOperand::new(context))))
    }
}

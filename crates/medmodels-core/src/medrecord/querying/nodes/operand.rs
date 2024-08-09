#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::operation::{NodeOperation, NodeValueOperation, NodeValuesOperation};
use crate::{
    medrecord::{
        querying::{
            edges::{EdgeOperand, EdgeOperation},
            evaluate::{EvaluateOperand, EvaluateOperation},
            wrapper::{CardinalityWrapper, DeepClone, Wrapper},
        },
        Group, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Clone)]
pub struct NodeOperand {
    operations: Vec<NodeOperation>,
}

impl EvaluateOperand for NodeOperand {
    type Index = NodeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let node_indices =
            Box::new(medrecord.node_indices()) as Box<dyn Iterator<Item = &'a NodeIndex>>;

        self.operations
            .iter()
            .fold(Box::new(node_indices), |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }
}

impl DeepClone for NodeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl NodeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.operations.push(NodeOperation::InGroup {
            group: group.into(),
        });
    }

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        let mut operand = EdgeOperand::new();

        let context = EdgeOperation::OutgoingEdgesContext {
            context: self.deep_clone(),
        };

        operand.operations.push(context);

        let operand = Wrapper::from(operand);

        self.operations.push(NodeOperation::OutgoingEdges {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<NodeOperand> {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(NodeOperand::new())))
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.0.borrow_mut().in_group(group);
    }

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        self.0.borrow_mut().outgoing_edges()
    }
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
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        self.context.evaluate(medrecord)
        // TODO
    }
}

impl DeepClone for NodeValuesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            attribute: self.attribute.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
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
pub struct NodeValueOperand {
    context: Wrapper<NodeValuesOperand>,
    operations: Vec<NodeValueOperation>,
}

impl DeepClone for NodeValueOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
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

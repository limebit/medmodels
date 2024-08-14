#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::operation::{NodeOperation, NodeValueOperation, NodeValuesOperation};
use crate::{
    medrecord::{
        querying::{
            edges::EdgeOperand,
            traits::{DeepClone, EvaluateOperand, EvaluateOperation, ReadWriteOrPanic},
            wrapper::{CardinalityWrapper, Wrapper},
        },
        Group, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::fmt::Debug;

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

    pub fn has_attribute<A>(&mut self, attribute: A)
    where
        A: Into<CardinalityWrapper<MedRecordAttribute>>,
    {
        self.operations.push(NodeOperation::HasAttribute {
            attribute: attribute.into(),
        });
    }

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        let operand = Wrapper::<EdgeOperand>::new();

        self.operations.push(NodeOperation::OutgoingEdges {
            operand: operand.clone(),
        });

        operand
    }

    pub fn incoming_edges(&mut self) -> Wrapper<EdgeOperand> {
        let operand = Wrapper::<EdgeOperand>::new();

        self.operations.push(NodeOperation::IncomingEdges {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<NodeOperand> {
    pub(crate) fn new() -> Self {
        NodeOperand::new().into()
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.0.write_or_panic().in_group(group);
    }

    pub fn has_attribute<A>(&mut self, attribute: A)
    where
        A: Into<CardinalityWrapper<MedRecordAttribute>>,
    {
        self.0.write_or_panic().has_attribute(attribute);
    }

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        self.0.write_or_panic().outgoing_edges()
    }

    pub fn incoming_edges(&mut self) -> Wrapper<EdgeOperand> {
        self.0.write_or_panic().incoming_edges()
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
        let node_indices = self.context.evaluate(medrecord);

        self.operations
            .iter()
            .fold(Box::new(node_indices), |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
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
        NodeValuesOperand::new(context, attribute).into()
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
        NodeValueOperand::new(context).into()
    }
}

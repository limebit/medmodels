use super::operation::NodeOperation;
use crate::{
    medrecord::{
        querying::{
            edges::EdgeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::{Context, MedRecordValuesOperand},
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        let node_indices =
            Box::new(medrecord.node_indices()) as Box<dyn Iterator<Item = &'a NodeIndex>>;

        self.operations
            .iter()
            .fold(Box::new(node_indices), |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MedRecordValuesOperand> {
        let operand = Wrapper::<MedRecordValuesOperand>::new(
            Context::NodeOperand(self.deep_clone()),
            attribute,
        );

        self.operations.push(NodeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MedRecordValuesOperand> {
        self.0.write_or_panic().attribute(attribute)
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

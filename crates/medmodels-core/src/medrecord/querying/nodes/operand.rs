use super::NodeOperation;
use crate::{
    medrecord::{
        querying::{
            edges::{EdgeOperandWrapper, EdgeOperation},
            wrapper::Wrapper,
        },
        Group, NodeIndex,
    },
    MedRecord,
};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Clone)]
pub struct NodeOperand {
    operations: Vec<NodeOperation>,
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
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let node_indices =
            Box::new(medrecord.node_indices()) as Box<dyn Iterator<Item = &'a NodeIndex>>;

        self.operations
            .iter()
            .fold(node_indices, |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<Wrapper<Group>>,
    {
        self.operations.push(NodeOperation::InGroup {
            group: group.into(),
        });
    }

    pub fn outgoing_edges(&mut self) -> EdgeOperandWrapper {
        let operand = EdgeOperandWrapper::new();

        let context = EdgeOperation::OutgoingEdgesContext {
            operand: self.clone().into(),
        };

        operand.0.borrow_mut().operations.push(context);

        self.operations.push(NodeOperation::OutgoingEdges {
            operand: operand.clone(),
        });

        operand
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct NodeOperandWrapper(pub(crate) Rc<RefCell<NodeOperand>>);

impl From<NodeOperand> for NodeOperandWrapper {
    fn from(value: NodeOperand) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}

impl NodeOperandWrapper {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(NodeOperand::new())))
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<Wrapper<Group>>,
    {
        self.0.borrow_mut().in_group(group);
    }

    pub fn outgoing_edges(&mut self) -> EdgeOperandWrapper {
        self.0.borrow_mut().outgoing_edges()
    }
}

use super::NodeOperation;
use crate::{
    medrecord::{
        querying::{
            edges::{EdgeOperand, EdgeOperation},
            evaluate::{EvaluateOperand, EvaluateOperation},
            wrapper::{CardinalityWrapper, OperandContext, Wrapper},
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

impl EvaluateOperand for NodeOperand {
    type Index = NodeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let node_indices =
            Box::new(medrecord.node_indices()) as Box<dyn Iterator<Item = &'a NodeIndex>>;

        self.operations[0..end_index.unwrap_or(self.operations.len())]
            .iter()
            .fold(Box::new(node_indices), |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
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

    pub fn outgoing_edges(&mut self, self_wrapper: &Wrapper<NodeOperand>) -> Wrapper<EdgeOperand> {
        let mut operand = EdgeOperand::new();

        let context = EdgeOperation::OutgoingEdgesContext {
            context: OperandContext::new(self_wrapper.clone(), self.operations.len()),
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
        self.0.borrow_mut().outgoing_edges(self)
    }
}

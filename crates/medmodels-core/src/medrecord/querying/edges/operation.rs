use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperandContext, EvaluateOperation},
            nodes::NodeOperand,
            wrapper::{OperandContext, Wrapper},
        },
        EdgeIndex,
    },
    MedRecord,
};
use std::collections::HashSet;

use super::values::EdgeValuesOperand;

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    // If this operation is used, it is always the first operation of an operand.
    OutgoingEdgesContext {
        context: OperandContext<NodeOperand>,
    },

    ConnectsTo {
        operand: Wrapper<NodeOperand>,
    },

    Attribute {
        operand: Wrapper<EdgeValuesOperand>,
    },
}

impl EvaluateOperation for EdgeOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::OutgoingEdgesContext { context } => Box::new(
                Self::evaluate_outgoing_edges_context(medrecord, context.clone()),
            ),
            Self::ConnectsTo { operand } => Box::new(Self::evaluate_connects_to(
                medrecord,
                indices,
                operand.clone(),
            )),
            Self::Attribute { operand } => {
                Box::new(Self::evaluate_attribute(medrecord, operand.clone()))
            }
        }
    }
}

impl EdgeOperation {
    fn evaluate_outgoing_edges_context(
        medrecord: &MedRecord,
        context: OperandContext<NodeOperand>,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let node_indices = context.evaluate(medrecord);

        node_indices.flat_map(|node_index| {
            let outgoing_edges = medrecord
                .outgoing_edges(node_index)
                .expect("Node must exist");

            outgoing_edges
        })
    }

    fn evaluate_connects_to<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: Wrapper<NodeOperand>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_indices = operand.evaluate(medrecord, None).collect::<HashSet<_>>();

        edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        })
    }

    fn evaluate_attribute(
        medrecord: &MedRecord,
        operand: Wrapper<EdgeValuesOperand>,
    ) -> impl Iterator<Item = &EdgeIndex> {
        operand.evaluate(medrecord, None)
    }
}

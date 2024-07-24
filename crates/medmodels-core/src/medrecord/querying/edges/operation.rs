use crate::{
    medrecord::{querying::nodes::NodeOperandWrapper, EdgeIndex},
    MedRecord,
};
use std::collections::HashSet;

use super::operand::EdgeValuesOperandWrapper;

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    // If this operation is used, it is always the first operation of an operand.
    OutgoingEdgesContext { operand: NodeOperandWrapper },
    ConnectsTo { operand: NodeOperandWrapper },

    Attribute { operand: EdgeValuesOperandWrapper },
}

impl EdgeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        match self {
            Self::OutgoingEdgesContext { operand } => Box::new(
                Self::evaluate_outgoing_edges_context(medrecord, operand.clone()),
            ),
            Self::ConnectsTo { operand } => Box::new(Self::evaluate_connects_to(
                medrecord,
                edge_indices,
                operand.clone(),
            )),
            Self::Attribute { operand } => Box::new(Self::evaluate_attribute(
                medrecord,
                edge_indices,
                operand.clone(),
            )),
        }
    }

    fn evaluate_outgoing_edges_context(
        medrecord: &MedRecord,
        operand: NodeOperandWrapper,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let node_indices = operand.0.borrow().evaluate(medrecord);

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
        operand: NodeOperandWrapper,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_indices = operand
            .0
            .borrow()
            .evaluate(medrecord)
            .collect::<HashSet<_>>();

        edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        })
    }

    fn evaluate_attribute<'a>(
        _medrecord: &'a MedRecord,
        _edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        _operand: EdgeValuesOperandWrapper,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        // TODO
        Vec::new().into_iter()
    }
}

#[derive(Debug, Clone)]
pub enum EdgeValuesOperation {}

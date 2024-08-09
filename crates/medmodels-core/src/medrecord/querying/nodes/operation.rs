#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::NodeValueOperand;
use crate::medrecord::{
    querying::{
        edges::EdgeOperand,
        evaluate::{EvaluateOperand, EvaluateOperation},
        values::ComparisonOperand,
        wrapper::{CardinalityWrapper, Wrapper},
    },
    Group, MedRecord, NodeIndex,
};
use roaring::RoaringBitmap;

#[derive(Debug, Clone)]
pub enum NodeOperation {
    InGroup { group: CardinalityWrapper<Group> },
    OutgoingEdges { operand: Wrapper<EdgeOperand> },
}

impl EvaluateOperation for NodeOperation {
    type Index = NodeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::InGroup { group } => {
                Box::new(Self::evaluate_in_group(medrecord, indices, group.clone()))
            }
            Self::OutgoingEdges { operand } => Box::new(Self::evaluate_outgoing_edges(
                medrecord,
                indices,
                operand.clone(),
            )),
        }
    }
}

impl NodeOperation {
    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        group: CardinalityWrapper<Group>,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        node_indices.filter(move |node_index| {
            let groups_of_node = medrecord
                .groups_of_node(node_index)
                .expect("Node must exist");

            let groups_of_node = groups_of_node.collect::<Vec<_>>();

            match &group {
                CardinalityWrapper::Single(group) => groups_of_node.contains(&group),
                CardinalityWrapper::Multiple(groups) => {
                    groups.iter().all(|group| groups_of_node.contains(&group))
                }
            }
        })
    }

    fn evaluate_outgoing_edges<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: Wrapper<EdgeOperand>,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        let edge_indices = operand.evaluate(medrecord).collect::<RoaringBitmap>();

        node_indices.filter(move |node_index| {
            let outgoing_edge_indices = medrecord
                .outgoing_edges(node_index)
                .expect("Node must exist");

            let outgoing_edge_indices = outgoing_edge_indices.collect::<RoaringBitmap>();

            !outgoing_edge_indices.is_disjoint(&edge_indices)
        })
    }
}

#[derive(Debug, Clone)]
pub enum NodeValuesOperation {
    Max { operand: Wrapper<NodeValueOperand> },
}

#[derive(Debug, Clone)]
pub enum NodeValueOperation {
    LessThan { operand: ComparisonOperand },
}

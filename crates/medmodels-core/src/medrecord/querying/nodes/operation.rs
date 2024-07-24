use crate::{
    medrecord::{
        querying::{edges::EdgeOperandWrapper, wrapper::Wrapper},
        Group, NodeIndex,
    },
    MedRecord,
};
use roaring::RoaringBitmap;

#[derive(Debug, Clone)]
pub enum NodeOperation {
    InGroup { group: Wrapper<Group> },
    OutgoingEdges { operand: EdgeOperandWrapper },
}

impl NodeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        match self {
            NodeOperation::InGroup { group } => Box::new(Self::evaluate_in_group(
                medrecord,
                node_indices,
                group.clone(),
            )),
            NodeOperation::OutgoingEdges { operand } => Box::new(Self::evaluate_outgoing_edges(
                medrecord,
                node_indices,
                operand.clone(),
            )),
        }
    }

    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        group: Wrapper<Group>,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        node_indices.filter(move |node_index| {
            let groups_of_node = medrecord
                .groups_of_node(node_index)
                .expect("Node must exist");

            let groups_of_node = groups_of_node.collect::<Vec<_>>();

            match &group {
                Wrapper::Single(group) => groups_of_node.contains(&group),
                Wrapper::Multiple(groups) => {
                    groups.iter().all(|group| groups_of_node.contains(&group))
                }
            }
        })
    }

    fn evaluate_outgoing_edges<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: EdgeOperandWrapper,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        let edge_indices = operand
            .0
            .borrow()
            .evaluate(medrecord)
            .collect::<RoaringBitmap>();

        node_indices.filter(move |node_index| {
            let outgoing_edge_indices = medrecord
                .outgoing_edges(node_index)
                .expect("Node must exist");

            let outgoing_edge_indices = outgoing_edge_indices.collect::<RoaringBitmap>();

            !outgoing_edge_indices.is_disjoint(&edge_indices)
        })
    }
}

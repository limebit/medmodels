use crate::medrecord::{
    querying::{
        edges::EdgeOperand,
        traits::{DeepClone, ReadWriteOrPanic},
        values::MedRecordValuesOperand,
        wrapper::{CardinalityWrapper, Wrapper},
    },
    Group, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex,
};
use roaring::RoaringBitmap;

#[derive(Debug, Clone)]
pub enum NodeOperation {
    Attribute {
        operand: Wrapper<MedRecordValuesOperand>,
    },

    InGroup {
        group: CardinalityWrapper<Group>,
    },
    HasAttribute {
        attribute: CardinalityWrapper<MedRecordAttribute>,
    },

    OutgoingEdges {
        operand: Wrapper<EdgeOperand>,
    },
    IncomingEdges {
        operand: Wrapper<EdgeOperand>,
    },
}

impl DeepClone for NodeOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Attribute { operand } => Self::Attribute {
                operand: operand.deep_clone(),
            },
            Self::InGroup { group } => Self::InGroup {
                group: group.clone(),
            },
            Self::HasAttribute { attribute } => Self::HasAttribute {
                attribute: attribute.clone(),
            },
            Self::OutgoingEdges { operand } => Self::OutgoingEdges {
                operand: operand.deep_clone(),
            },
            Self::IncomingEdges { operand } => Self::IncomingEdges {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl NodeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        match self {
            Self::Attribute { operand } => Box::new(Self::evaluate_attribute(
                medrecord,
                node_indices,
                operand.clone(),
            )),
            Self::InGroup { group } => Box::new(Self::evaluate_in_group(
                medrecord,
                node_indices,
                group.clone(),
            )),
            Self::HasAttribute { attribute } => Box::new(Self::evaluate_has_attribute(
                medrecord,
                node_indices,
                attribute.clone(),
            )),
            Self::OutgoingEdges { operand } => Box::new(Self::evaluate_outgoing_edges(
                medrecord,
                node_indices,
                operand.clone(),
            )),
            Self::IncomingEdges { operand } => Box::new(Self::evaluate_incoming_edges(
                medrecord,
                node_indices,
                operand.clone(),
            )),
        }
    }

    #[inline]
    pub(crate) fn get_values<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = (&'a NodeIndex, &'a MedRecordValue)> + 'a {
        node_indices.flat_map(move |node_index| {
            Some((
                node_index,
                medrecord
                    .node_attributes(node_index)
                    .expect("Edge must exist")
                    .get(&attribute)?,
            ))
        })
    }

    #[inline]
    fn evaluate_attribute<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: Wrapper<MedRecordValuesOperand>,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let values = Self::get_values(
            medrecord,
            node_indices,
            operand.0.read_or_panic().attribute.clone(),
        );

        operand.evaluate(&medrecord, values).map(|value| value.0)
    }

    #[inline]
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

    #[inline]
    fn evaluate_has_attribute<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        attribute: CardinalityWrapper<MedRecordAttribute>,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        node_indices.filter(move |node_index| {
            let attributes_of_node = medrecord
                .node_attributes(node_index)
                .expect("Node must exist")
                .keys();

            let attributes_of_node = attributes_of_node.collect::<Vec<_>>();

            match &attribute {
                CardinalityWrapper::Single(attribute) => attributes_of_node.contains(&attribute),
                CardinalityWrapper::Multiple(attributes) => attributes
                    .iter()
                    .all(|attribute| attributes_of_node.contains(&attribute)),
            }
        })
    }

    #[inline]
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

    #[inline]
    fn evaluate_incoming_edges<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: Wrapper<EdgeOperand>,
    ) -> impl Iterator<Item = &'a NodeIndex> + 'a {
        let edge_indices = operand.evaluate(medrecord).collect::<RoaringBitmap>();

        node_indices.filter(move |node_index| {
            let incoming_edge_indices = medrecord
                .incoming_edges(node_index)
                .expect("Node must exist");

            let incoming_edge_indices = incoming_edge_indices.collect::<RoaringBitmap>();

            !incoming_edge_indices.is_disjoint(&edge_indices)
        })
    }
}

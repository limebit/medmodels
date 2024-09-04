use crate::{
    medrecord::{
        querying::{
            nodes::NodeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::MedRecordValuesOperand,
            wrapper::Wrapper,
        },
        CardinalityWrapper, EdgeIndex, Group, MedRecordAttribute, MedRecordValue,
    },
    MedRecord,
};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Attribute {
        operand: Wrapper<MedRecordValuesOperand>,
    },

    InGroup {
        group: CardinalityWrapper<Group>,
    },
    HasAttribute {
        attribute: CardinalityWrapper<MedRecordAttribute>,
    },

    SourceNode {
        operand: Wrapper<NodeOperand>,
    },
    TargetNode {
        operand: Wrapper<NodeOperand>,
    },
}

impl DeepClone for EdgeOperation {
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
            Self::SourceNode { operand } => Self::SourceNode {
                operand: operand.deep_clone(),
            },
            Self::TargetNode { operand } => Self::TargetNode {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl EdgeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        match self {
            Self::Attribute { operand } => Box::new(Self::evaluate_attribute(
                medrecord,
                edge_indices,
                operand.clone(),
            )),
            Self::InGroup { group } => Box::new(Self::evaluate_in_group(
                medrecord,
                edge_indices,
                group.clone(),
            )),
            Self::HasAttribute { attribute } => Box::new(Self::evaluate_has_attribute(
                medrecord,
                edge_indices,
                attribute.clone(),
            )),
            Self::SourceNode { operand } => {
                Box::new(Self::evaluate_source_node(medrecord, edge_indices, operand))
            }
            Self::TargetNode { operand } => {
                Box::new(Self::evaluate_target_node(medrecord, edge_indices, operand))
            }
        }
    }

    #[inline]
    pub(crate) fn get_values<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = (&'a EdgeIndex, &'a MedRecordValue)> + 'a {
        edge_indices.flat_map(move |edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?,
            ))
        })
    }

    #[inline]
    fn evaluate_attribute<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: Wrapper<MedRecordValuesOperand>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let values = Self::get_values(
            medrecord,
            edge_indices,
            operand.0.read_or_panic().attribute.clone(),
        );

        operand.evaluate(&medrecord, values).map(|value| value.0)
    }

    #[inline]
    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        group: CardinalityWrapper<Group>,
    ) -> impl Iterator<Item = &'a EdgeIndex> + 'a {
        edge_indices.filter(move |edge_index| {
            let groups_of_edge = medrecord
                .groups_of_edge(edge_index)
                .expect("Node must exist");

            let groups_of_edge = groups_of_edge.collect::<Vec<_>>();

            match &group {
                CardinalityWrapper::Single(group) => groups_of_edge.contains(&group),
                CardinalityWrapper::Multiple(groups) => {
                    groups.iter().all(|group| groups_of_edge.contains(&group))
                }
            }
        })
    }

    #[inline]
    fn evaluate_has_attribute<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        attribute: CardinalityWrapper<MedRecordAttribute>,
    ) -> impl Iterator<Item = &'a EdgeIndex> + 'a {
        edge_indices.filter(move |edge_index| {
            let attributes_of_edge = medrecord
                .edge_attributes(edge_index)
                .expect("Node must exist")
                .keys();

            let attributes_of_edge = attributes_of_edge.collect::<Vec<_>>();

            match &attribute {
                CardinalityWrapper::Single(attribute) => attributes_of_edge.contains(&attribute),
                CardinalityWrapper::Multiple(attributes) => attributes
                    .iter()
                    .all(|attribute| attributes_of_edge.contains(&attribute)),
            }
        })
    }

    #[inline]
    fn evaluate_source_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<NodeOperand>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_indices = operand.evaluate(medrecord).collect::<HashSet<_>>();

        edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        })
    }

    #[inline]
    fn evaluate_target_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<NodeOperand>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_indices = operand.evaluate(medrecord).collect::<HashSet<_>>();

        edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        })
    }
}

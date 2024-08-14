use super::{EdgeValueOperand, EdgeValuesOperand};
use crate::{
    medrecord::{
        querying::{
            nodes::NodeOperand,
            traits::{DeepClone, EvaluateOperand, EvaluateOperation, ReadWriteOrPanic},
            values::{ComparisonOperand, ValueKind, ValuesOperand},
            wrapper::Wrapper,
        },
        CardinalityWrapper, EdgeIndex, Group, MedRecordAttribute,
    },
    MedRecord,
};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Attribute {
        operand: Wrapper<EdgeValuesOperand>,
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

impl EvaluateOperation for EdgeOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::Attribute { operand } => {
                Box::new(Self::evaluate_attribute(medrecord, operand.clone()))
            }
            Self::InGroup { group } => {
                Box::new(Self::evaluate_in_group(medrecord, indices, group.clone()))
            }
            Self::HasAttribute { attribute } => Box::new(Self::evaluate_has_attribute(
                medrecord,
                indices,
                attribute.clone(),
            )),
            Self::SourceNode { operand } => Box::new(Self::evaluate_source_node(
                medrecord,
                indices,
                operand.clone(),
            )),
            Self::TargetNode { operand } => Box::new(Self::evaluate_target_node(
                medrecord,
                indices,
                operand.clone(),
            )),
        }
    }
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
    fn evaluate_attribute(
        medrecord: &MedRecord,
        operand: Wrapper<EdgeValuesOperand>,
    ) -> impl Iterator<Item = &EdgeIndex> {
        operand.evaluate(medrecord)
    }

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

    fn evaluate_source_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: Wrapper<NodeOperand>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_indices = operand.evaluate(medrecord).collect::<HashSet<_>>();

        edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        })
    }

    fn evaluate_target_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: Wrapper<NodeOperand>,
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

#[derive(Debug, Clone)]
pub enum EdgeValuesOperation {
    Max { operand: Wrapper<EdgeValueOperand> },
}

impl EvaluateOperation for EdgeValuesOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        _indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::Max { operand } => Box::new(Self::evaluate_max(medrecord, operand.clone())),
        }
    }
}

impl DeepClone for EdgeValuesOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Max { operand } => Self::Max {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl EdgeValuesOperation {
    fn evaluate_max(
        medrecord: &MedRecord,
        operand: Wrapper<EdgeValueOperand>,
    ) -> impl Iterator<Item = &EdgeIndex> {
        operand.evaluate(medrecord)
    }
}

#[derive(Debug, Clone)]
pub enum EdgeValueOperation {
    LessThan {
        operand: ComparisonOperand,
        kind: ValueKind,
    },
}

impl EvaluateOperation for EdgeValueOperation {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        match self {
            Self::LessThan { operand, kind } => Box::new(Self::evaluate_less_than(
                medrecord,
                indices,
                operand.clone(),
                kind.clone(),
            )),
        }
    }
}

impl DeepClone for EdgeValueOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::LessThan { operand, kind } => Self::LessThan {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
        }
    }
}

impl EdgeValueOperation {
    fn evaluate_less_than<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: ComparisonOperand,
        kind: ValueKind,
    ) -> impl Iterator<Item = &EdgeIndex> {
        match kind {
            ValueKind::Max(values) => {
                let values = values.evaluate_edge_values(medrecord, edge_indices);
            }
            ValueKind::Min(_) => todo!(),

            ValueKind::All(_) => todo!(),
            ValueKind::Any(_) => todo!(),
        }

        edge_indices
    }
}

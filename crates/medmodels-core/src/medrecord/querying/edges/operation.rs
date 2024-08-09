use super::{EdgeValueOperand, EdgeValuesOperand};
use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperandContext, EvaluateOperation},
            nodes::NodeOperand,
            values::{ComparisonOperand, ValuesOperand},
            wrapper::{OperandContext, Wrapper},
        },
        EdgeIndex, MedRecordAttribute,
    },
    MedRecord,
};
use std::collections::HashSet;

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
        let node_indices = operand.evaluate(medrecord).collect::<HashSet<_>>();

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
        operand.evaluate(medrecord)
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
    // If this operation is used, it is always the first operation of an operand.
    MaxContext {
        context: OperandContext<EdgeValuesOperand>,
        attribute: MedRecordAttribute,
    },

    LessThan {
        operand: ComparisonOperand,
        attribute: MedRecordAttribute,
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
            Self::MaxContext { context, attribute } => Box::new(Self::evaluate_max_context(
                medrecord,
                context.clone(),
                attribute.clone(),
            )),
            Self::LessThan { operand, attribute } => Box::new(Self::evaluate_less_than(
                medrecord,
                indices,
                operand.clone(),
                attribute.clone(),
            )),
        }
    }
}

impl EdgeValueOperation {
    fn evaluate_max_context(
        medrecord: &MedRecord,
        context: OperandContext<EdgeValuesOperand>,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let edge_indices = context.evaluate(medrecord);

        let mut edge_attributes = edge_indices.filter_map(|edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?,
            ))
        });

        let Some(max) = edge_attributes.next() else {
            return Vec::new().into_iter();
        };

        let max_edge =
            edge_attributes.fold(max, |max, edge| if edge.1 > max.1 { edge } else { max });

        vec![max_edge.0].into_iter()
    }

    fn evaluate_less_than<'a>(
        medrecord: &'a MedRecord,
        mut edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: ComparisonOperand,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let Some(edge_index) = edge_indices.next() else {
            return Vec::new().into_iter();
        };
        let value = medrecord
            .edge_attributes(edge_index)
            .expect("Edge must exist")
            .get(&attribute)
            .expect("Attribute must exist");

        let ComparisonOperand::Multiple(comparison) = operand else {
            todo!()
        };

        let ValuesOperand::Edges(operand) = comparison else {
            todo!()
        };

        let comparison_edge_indices = operand.evaluate(medrecord);
        let comparison_attribute = operand.0.borrow().attribute.clone();

        let comparison_values = comparison_edge_indices.filter_map(|edge_index| {
            medrecord
                .edge_attributes(edge_index)
                .expect("Edge must exist")
                .get(&comparison_attribute)
        });

        for comparison_value in comparison_values {
            if value >= comparison_value {
                return Vec::new().into_iter();
            }
        }

        vec![edge_index].into_iter()
    }
}

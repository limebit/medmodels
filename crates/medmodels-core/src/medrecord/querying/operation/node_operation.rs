use super::{
    edge_operation::EdgeOperation,
    operand::{IntoVecNodeIndex, NodeIndexInOperand},
    AttributeOperation, Operation,
};
use crate::medrecord::{
    datatypes::{Contains, EndsWith, StartsWith},
    MedRecord, MedRecordAttribute, NodeIndex,
};

macro_rules! implement_index_evaluate {
    ($name: ident, $evaluate: ident) => {
        fn $name<'a>(
            indices: impl Iterator<Item = &'a NodeIndex>,
            operand: NodeIndex,
        ) -> impl Iterator<Item = &'a NodeIndex> {
            indices.filter(move |index| (*index).$evaluate(&operand))
        }
    };
}

#[derive(Debug, Clone)]
pub enum NodeIndexOperation {
    Gt(NodeIndex),
    Lt(NodeIndex),
    Gte(NodeIndex),
    Lte(NodeIndex),
    Eq(NodeIndex),
    In(Box<NodeIndexInOperand>),
    StartsWith(NodeIndex),
    EndsWith(NodeIndex),
    Contains(NodeIndex),
}

#[derive(Debug, Clone)]
pub enum NodeOperation {
    Attribute(AttributeOperation),
    Index(NodeIndexOperation),

    InGroup(MedRecordAttribute),
    HasAttribute(MedRecordAttribute),

    HasIncomingEdgeWith(Box<EdgeOperation>),
    HasOutgoingEdgeWith(Box<EdgeOperation>),
    HasNeighborWith(Box<NodeOperation>),

    And(Box<(NodeOperation, NodeOperation)>),
    Or(Box<(NodeOperation, NodeOperation)>),
    Not(Box<NodeOperation>),
}

impl Operation for NodeOperation {
    type IndexType = NodeIndex;

    fn evaluate<'a>(
        self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::IndexType> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::IndexType> + 'a> {
        match self {
            NodeOperation::Attribute(attribute_operation) => {
                Self::evaluate_attribute(indices, attribute_operation, |index| {
                    medrecord.node_attributes(index)
                })
            }
            NodeOperation::Index(index_operation) => {
                Self::evaluate_index(medrecord, indices, index_operation)
            }

            NodeOperation::InGroup(attribute_operand) => Box::new(Self::evaluate_in_group(
                medrecord,
                indices,
                attribute_operand,
            )),
            NodeOperation::HasAttribute(attribute_operand) => Box::new(
                Self::evaluate_has_attribute(indices, attribute_operand, |index| {
                    medrecord.node_attributes(index)
                }),
            ),

            NodeOperation::HasOutgoingEdgeWith(operation) => Box::new(
                Self::evaluate_has_outgoing_edge_with(medrecord, indices, *operation),
            ),
            NodeOperation::HasIncomingEdgeWith(operation) => Box::new(
                Self::evaluate_has_incoming_edge_with(medrecord, indices, *operation),
            ),
            NodeOperation::HasNeighborWith(operation) => Box::new(
                Self::evaluate_has_neighbor_with(medrecord, indices, *operation),
            ),

            NodeOperation::And(operations) => Box::new(Self::evaluate_and(
                medrecord,
                indices.collect::<Vec<_>>(),
                (*operations).0,
                (*operations).1,
            )),
            NodeOperation::Or(operations) => Box::new(Self::evaluate_or(
                medrecord,
                indices.collect::<Vec<_>>(),
                (*operations).0,
                (*operations).1,
            )),
            NodeOperation::Not(operation) => Box::new(Self::evaluate_not(
                medrecord,
                indices.collect::<Vec<_>>(),
                *operation,
            )),
        }
    }
}

impl NodeOperation {
    pub fn and(self, operation: NodeOperation) -> NodeOperation {
        NodeOperation::And(Box::new((self, operation)))
    }

    pub fn or(self, operation: NodeOperation) -> NodeOperation {
        NodeOperation::Or(Box::new((self, operation)))
    }

    pub fn xor(self, operation: NodeOperation) -> NodeOperation {
        NodeOperation::And(Box::new((self, operation))).not()
    }

    pub fn not(self) -> NodeOperation {
        NodeOperation::Not(Box::new(self))
    }

    fn evaluate_index<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operation: NodeIndexOperation,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        match operation {
            NodeIndexOperation::Gt(operand) => {
                Box::new(Self::evaluate_index_gt(node_indices, operand))
            }
            NodeIndexOperation::Lt(operand) => {
                Box::new(Self::evaluate_index_lt(node_indices, operand))
            }
            NodeIndexOperation::Gte(operand) => {
                Box::new(Self::evaluate_index_gte(node_indices, operand))
            }
            NodeIndexOperation::Lte(operand) => {
                Box::new(Self::evaluate_index_lte(node_indices, operand))
            }
            NodeIndexOperation::Eq(operand) => {
                Box::new(Self::evaluate_index_eq(node_indices, operand))
            }
            NodeIndexOperation::In(operands) => Box::new(Self::evaluate_index_in(
                node_indices,
                operands.into_vec_node_index(medrecord),
            )),
            NodeIndexOperation::StartsWith(operand) => {
                Box::new(Self::evaluate_index_starts_with(node_indices, operand))
            }
            NodeIndexOperation::EndsWith(operand) => {
                Box::new(Self::evaluate_index_ends_with(node_indices, operand))
            }
            NodeIndexOperation::Contains(operand) => {
                Box::new(Self::evaluate_index_contains(node_indices, operand))
            }
        }
    }

    implement_index_evaluate!(evaluate_index_starts_with, starts_with);
    implement_index_evaluate!(evaluate_index_ends_with, ends_with);
    implement_index_evaluate!(evaluate_index_contains, contains);

    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute_operand: MedRecordAttribute,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let nodes_in_group = match medrecord.nodes_in_group(&attribute_operand) {
            Ok(nodes_in_group) => nodes_in_group.collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };

        node_indices.filter(move |index| nodes_in_group.contains(index))
    }

    fn evaluate_has_outgoing_edge_with<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operation: EdgeOperation,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices.filter(move |index| {
            let Ok(edges) = medrecord.outgoing_edges(index) else {
                return false;
            };

            let edge_indices = operation.clone().evaluate(medrecord, edges);

            edge_indices.count() > 0
        })
    }

    fn evaluate_has_incoming_edge_with<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operation: EdgeOperation,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices.filter(move |index| {
            let Ok(edges) = medrecord.incoming_edges(index) else {
                return false;
            };

            operation.clone().evaluate(medrecord, edges).count() > 0
        })
    }

    fn evaluate_has_neighbor_with<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operation: NodeOperation,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices.filter(move |index| {
            let Ok(neighbors) = medrecord.neighbors(index) else {
                return false;
            };

            operation.clone().evaluate(medrecord, neighbors).count() > 0
        })
    }
}

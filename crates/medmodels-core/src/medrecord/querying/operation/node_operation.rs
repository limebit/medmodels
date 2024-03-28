use super::{
    edge_operation::EdgeOperation, operand::NodeAttributeOperand, AttributeOperation, Operation,
};
use crate::medrecord::{MedRecord, NodeIndex};

#[derive(Debug, Clone)]
pub enum NodeIndexOperation {
    Gt(NodeIndex),
    Gte(NodeIndex),
    Eq(NodeIndex),
    In(Vec<NodeIndex>),
    StartsWith(NodeIndex),
    EndsWith(NodeIndex),
    Contains(NodeIndex),
}

#[derive(Debug, Clone)]
pub enum NodeOperation {
    Attribute(AttributeOperation<NodeAttributeOperand>),
    Index(NodeIndexOperation),

    ConnectedWith(NodeAttributeOperand, Box<EdgeOperation>),

    Connected(NodeAttributeOperand),
    InGroup(NodeAttributeOperand),
    HasAttribute(NodeAttributeOperand),

    And(Box<(NodeOperation, NodeOperation)>),
    Or(Box<(NodeOperation, NodeOperation)>),
    Not(Box<NodeOperation>),
}

impl Operation for NodeOperation {
    type IndexType = NodeIndex;
    type AttributeOperand = NodeAttributeOperand;

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
            NodeOperation::Index(index_operation) => Self::evaluate_index(indices, index_operation),

            NodeOperation::ConnectedWith(attribute_operand, operation) => Box::new(
                Self::evaluate_connected_with(medrecord, indices, attribute_operand, *operation),
            ),

            NodeOperation::Connected(attribute_operand) => Box::new(Self::evalaute_connected(
                medrecord,
                indices,
                attribute_operand,
            )),
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
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operation: NodeIndexOperation,
    ) -> Box<dyn Iterator<Item = &'a NodeIndex> + 'a> {
        match operation {
            NodeIndexOperation::Gt(operand) => {
                Box::new(Self::evaluate_index_gt(node_indices, operand))
            }
            NodeIndexOperation::Gte(operand) => {
                Box::new(Self::evaluate_index_gte(node_indices, operand))
            }
            NodeIndexOperation::Eq(operand) => {
                Box::new(Self::evaluate_index_eq(node_indices, operand))
            }
            NodeIndexOperation::In(operands) => {
                Box::new(Self::evaluate_index_in(node_indices, operands))
            }
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

    fn evaluate_index_starts_with<'a>(
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        _operand: NodeIndex,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices
    }

    fn evaluate_index_ends_with<'a>(
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        _operand: NodeIndex,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices
    }

    fn evaluate_index_contains<'a>(
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        _operand: NodeIndex,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices
    }

    fn evaluate_connected_with<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute_operand: NodeAttributeOperand,
        operation: EdgeOperation,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let node_index = attribute_operand.into();

        node_indices.filter(move |index| {
            let edges = medrecord.edges_connecting(index, &node_index);

            let edge_indices = operation.clone().evaluate(medrecord, edges); // TODO: check if clone is really necessary

            edge_indices.count() > 0
        })
    }

    fn evalaute_connected<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute_operand: NodeAttributeOperand,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let node_index = attribute_operand.into();

        node_indices.filter(move |index| {
            let edges = medrecord.edges_connecting(index, &node_index);

            edges.count() > 0
        })
    }

    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute_operand: NodeAttributeOperand,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        let nodes_in_group = match medrecord.nodes_in_group(&attribute_operand.into()) {
            Ok(nodes_in_group) => nodes_in_group.collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };

        node_indices.filter(move |index| nodes_in_group.contains(index))
    }
}

use super::{operand::EdgeAttributeOperand, AttributeOperation, Operation};
use crate::medrecord::{EdgeIndex, MedRecord};

#[derive(Debug, Clone)]
pub enum EdgeIndexOperation {
    Gt(EdgeIndex),
    Gte(EdgeIndex),
    Eq(EdgeIndex),
    In(Vec<EdgeIndex>),
}

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Attribute(AttributeOperation<EdgeAttributeOperand>),
    Index(EdgeIndexOperation),

    ConnectedSource(EdgeAttributeOperand),
    ConnectedTarget(EdgeAttributeOperand),
    HasAttribute(EdgeAttributeOperand),

    And(Box<(EdgeOperation, EdgeOperation)>),
    Or(Box<(EdgeOperation, EdgeOperation)>),
    Not(Box<EdgeOperation>),
}

impl Operation for EdgeOperation {
    type IndexType = EdgeIndex;
    type AttributeOperand = EdgeAttributeOperand;

    fn evaluate<'a>(
        self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::IndexType> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::IndexType> + 'a> {
        match self {
            EdgeOperation::Attribute(attribute_operation) => {
                Self::evaluate_attribute(indices, attribute_operation, |index| {
                    medrecord.edge_attributes(index)
                })
            }
            EdgeOperation::Index(index_operation) => Self::evaluate_index(indices, index_operation),

            EdgeOperation::ConnectedSource(attribute_operand) => Box::new(
                Self::evaluate_connected_to(medrecord, indices, attribute_operand),
            ),
            EdgeOperation::ConnectedTarget(attribute_operand) => Box::new(
                Self::evaluate_connected_from(medrecord, indices, attribute_operand),
            ),
            EdgeOperation::HasAttribute(attribute_operand) => Box::new(
                Self::evaluate_has_attribute(indices, attribute_operand, |index| {
                    medrecord.edge_attributes(index)
                }),
            ),

            EdgeOperation::And(operations) => Box::new(Self::evaluate_and(
                medrecord,
                indices.collect::<Vec<_>>(),
                (*operations).0,
                (*operations).1,
            )),
            EdgeOperation::Or(operations) => Box::new(Self::evaluate_or(
                medrecord,
                indices.collect::<Vec<_>>(),
                (*operations).0,
                (*operations).1,
            )),
            EdgeOperation::Not(operation) => Box::new(Self::evaluate_not(
                medrecord,
                indices.collect::<Vec<_>>(),
                *operation,
            )),
        }
    }
}

impl EdgeOperation {
    pub fn and(self, operation: EdgeOperation) -> EdgeOperation {
        EdgeOperation::And(Box::new((self, operation)))
    }

    pub fn or(self, operation: EdgeOperation) -> EdgeOperation {
        EdgeOperation::Or(Box::new((self, operation)))
    }

    pub fn xor(self, operation: EdgeOperation) -> EdgeOperation {
        EdgeOperation::And(Box::new((self, operation))).not()
    }

    pub fn not(self) -> EdgeOperation {
        EdgeOperation::Not(Box::new(self))
    }

    fn evaluate_index<'a>(
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operation: EdgeIndexOperation,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        match operation {
            EdgeIndexOperation::Gt(operand) => {
                Box::new(Self::evaluate_index_gt(edge_indices, operand))
            }
            EdgeIndexOperation::Gte(operand) => {
                Box::new(Self::evaluate_index_gte(edge_indices, operand))
            }
            EdgeIndexOperation::Eq(operand) => {
                Box::new(Self::evaluate_index_eq(edge_indices, operand))
            }
            EdgeIndexOperation::In(operands) => {
                Box::new(Self::evaluate_index_in(edge_indices, operands))
            }
        }
    }

    fn evaluate_connected_to<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute_operand: EdgeAttributeOperand,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_index = attribute_operand.into();

        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            *endpoints.1 == node_index
        })
    }

    fn evaluate_connected_from<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute_operand: EdgeAttributeOperand,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        let node_index = attribute_operand.into();

        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            *endpoints.0 == node_index
        })
    }
}

use super::{
    operand::{ArithmeticOperation, EdgeIndexInOperand, IntoVecEdgeIndex, ValueOperand},
    AttributeOperation, NodeOperation, Operation,
};
use crate::medrecord::{
    datatypes::{Ceil, Floor, Lowercase, Round, Slice, Trim, TrimEnd, TrimStart, Uppercase},
    EdgeIndex, MedRecord, MedRecordAttribute,
};

#[derive(Debug, Clone)]
pub enum EdgeIndexOperation {
    Gt(EdgeIndex),
    Lt(EdgeIndex),
    Gte(EdgeIndex),
    Lte(EdgeIndex),
    Eq(EdgeIndex),
    In(Box<EdgeIndexInOperand>),
}

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Attribute(AttributeOperation),
    Index(EdgeIndexOperation),

    ConnectedSource(MedRecordAttribute),
    ConnectedTarget(MedRecordAttribute),
    HasAttribute(MedRecordAttribute),

    ConnectedSourceWith(Box<NodeOperation>),
    ConnectedTargetWith(Box<NodeOperation>),

    HasParallelEdgesWith(Box<EdgeOperation>),
    HasParallelEdgesWithSelfComparison(Box<EdgeOperation>),

    And(Box<(EdgeOperation, EdgeOperation)>),
    Or(Box<(EdgeOperation, EdgeOperation)>),
    Not(Box<EdgeOperation>),
}

impl Operation for EdgeOperation {
    type IndexType = EdgeIndex;

    fn evaluate<'a>(
        self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        match self {
            EdgeOperation::Attribute(attribute_operation) => {
                Self::evaluate_attribute(indices, attribute_operation, |index| {
                    medrecord.edge_attributes(index)
                })
            }
            EdgeOperation::Index(index_operation) => {
                Self::evaluate_index(medrecord, indices, index_operation)
            }

            EdgeOperation::ConnectedSource(attribute_operand) => Box::new(
                Self::evaluate_connected_target(medrecord, indices, attribute_operand),
            ),
            EdgeOperation::ConnectedTarget(attribute_operand) => Box::new(
                Self::evaluate_connected_source(medrecord, indices, attribute_operand),
            ),
            EdgeOperation::HasAttribute(attribute_operand) => Box::new(
                Self::evaluate_has_attribute(indices, attribute_operand, |index| {
                    medrecord.edge_attributes(index)
                }),
            ),

            EdgeOperation::ConnectedSourceWith(operation) => Box::new(
                Self::evaluate_connected_source_with(medrecord, indices, *operation),
            ),
            EdgeOperation::ConnectedTargetWith(operation) => Box::new(
                Self::evaluate_connected_target_with(medrecord, indices, *operation),
            ),

            EdgeOperation::HasParallelEdgesWith(operation) => {
                Self::evaluate_has_parallel_edges_with(medrecord, Box::new(indices), *operation)
            }
            EdgeOperation::HasParallelEdgesWithSelfComparison(operation) => {
                Self::evaluate_has_parallel_edges_with_compare_to_self(
                    medrecord,
                    Box::new(indices),
                    *operation,
                )
            }

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
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operation: EdgeIndexOperation,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        match operation {
            EdgeIndexOperation::Gt(operand) => {
                Box::new(Self::evaluate_index_gt(edge_indices, operand))
            }
            EdgeIndexOperation::Lt(operand) => {
                Box::new(Self::evaluate_index_lt(edge_indices, operand))
            }
            EdgeIndexOperation::Gte(operand) => {
                Box::new(Self::evaluate_index_gte(edge_indices, operand))
            }
            EdgeIndexOperation::Lte(operand) => {
                Box::new(Self::evaluate_index_lte(edge_indices, operand))
            }
            EdgeIndexOperation::Eq(operand) => {
                Box::new(Self::evaluate_index_eq(edge_indices, operand))
            }
            EdgeIndexOperation::In(operands) => Box::new(Self::evaluate_index_in(
                edge_indices,
                operands.into_vec_edge_index(medrecord),
            )),
        }
    }

    fn evaluate_connected_target<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute_operand: MedRecordAttribute,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            *endpoints.1 == attribute_operand
        })
    }

    fn evaluate_connected_source<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute_operand: MedRecordAttribute,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            *endpoints.0 == attribute_operand
        })
    }

    fn evaluate_connected_target_with<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operation: NodeOperation,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            operation
                .clone()
                .evaluate(medrecord, vec![endpoints.1].into_iter())
                .count()
                > 0
        })
    }

    fn evaluate_connected_source_with<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operation: NodeOperation,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            operation
                .clone()
                .evaluate(medrecord, vec![endpoints.0].into_iter())
                .count()
                > 0
        })
    }

    fn evaluate_has_parallel_edges_with<'a>(
        medrecord: &'a MedRecord,
        edge_indices: Box<dyn Iterator<Item = &'a EdgeIndex> + 'a>,
        operation: EdgeOperation,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        Box::new(edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            let edges = medrecord
                .edges_connecting(endpoints.0, endpoints.1)
                .filter(|other_index| other_index != index);

            operation.clone().evaluate(medrecord, edges).count() > 0
        }))
    }

    fn convert_value_operand<'a>(
        medrecord: &'a MedRecord,
        index: &'a EdgeIndex,
        value_operand: ValueOperand,
    ) -> Option<ValueOperand> {
        match value_operand {
            ValueOperand::Value(value) => Some(ValueOperand::Value(value)),
            ValueOperand::Evaluate(attribute) => Some(ValueOperand::Value(
                medrecord
                    .edge_attributes(index)
                    .ok()?
                    .get(&attribute)?
                    .clone(),
            )),
            ValueOperand::ArithmeticOperation(operation, attribute, other_value) => {
                let value = medrecord.edge_attributes(index).ok()?.get(&attribute)?;

                let result = match operation {
                    ArithmeticOperation::Addition => value.clone() + other_value,
                    ArithmeticOperation::Subtraction => value.clone() - other_value,
                    ArithmeticOperation::Multiplication => value.clone() * other_value,
                    ArithmeticOperation::Division => value.clone() / other_value,
                }
                .ok()?;

                Some(ValueOperand::Value(result))
            }
            ValueOperand::Slice(attribute, range) => {
                let value = medrecord.edge_attributes(index).ok()?.get(&attribute)?;

                Some(ValueOperand::Value(value.clone().slice(range)))
            }
            ValueOperand::TransformationOperation(operation, attribute) => {
                let value = medrecord.edge_attributes(index).ok()?.get(&attribute)?;

                let result = match operation {
                    super::operand::TransformationOperation::Round => value.clone().round(),
                    super::operand::TransformationOperation::Ceil => value.clone().ceil(),
                    super::operand::TransformationOperation::Floor => value.clone().floor(),
                    super::operand::TransformationOperation::Trim => value.clone().trim(),
                    super::operand::TransformationOperation::TrimStart => {
                        value.clone().trim_start()
                    }
                    super::operand::TransformationOperation::TrimEnd => value.clone().trim_end(),
                    super::operand::TransformationOperation::Lowercase => value.clone().lowercase(),
                    super::operand::TransformationOperation::Uppercase => value.clone().uppercase(),
                };

                Some(ValueOperand::Value(result))
            }
        }
    }
    fn evaluate_has_parallel_edges_with_compare_to_self<'a>(
        medrecord: &'a MedRecord,
        edge_indices: Box<dyn Iterator<Item = &'a EdgeIndex> + 'a>,
        operation: EdgeOperation,
    ) -> Box<dyn Iterator<Item = &'a EdgeIndex> + 'a> {
        Box::new(edge_indices.filter(move |index| {
            let Ok(endpoints) = medrecord.edge_endpoints(index) else {
                return false;
            };

            let edges = medrecord
                .edges_connecting(endpoints.0, endpoints.1)
                .filter(|other_index| other_index != index);

            let operation = operation.clone();

            let EdgeOperation::Attribute(operation) = operation else {
                return operation.evaluate(medrecord, edges).count() > 0;
            };

            match operation {
                AttributeOperation::Gt(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Gt(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Lt(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Lt(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Gte(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Gte(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Lte(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Lte(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Eq(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Eq(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Neq(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Neq(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::In(attribute, value) => {
                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::In(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::NotIn(attribute, value) => {
                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::In(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::StartsWith(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::StartsWith(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::EndsWith(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::EndsWith(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
                AttributeOperation::Contains(attribute, value) => {
                    let Some(value) = Self::convert_value_operand(medrecord, index, value) else {
                        return false;
                    };

                    Self::evaluate_attribute(
                        edges,
                        AttributeOperation::Contains(attribute, value),
                        |index| medrecord.edge_attributes(index),
                    )
                    .count()
                        > 0
                }
            }
        }))
    }
}

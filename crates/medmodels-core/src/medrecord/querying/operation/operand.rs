use super::{
    edge_operation::EdgeIndexOperation,
    node_operation::{NodeIndexOperation, NodeOperation},
    AttributeOperation, EdgeOperation, Operation,
};
use crate::medrecord::{EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex};
use std::{fmt::Debug, ops::Range};

#[derive(Debug, Clone)]
pub enum ArithmeticOperation {
    Addition,
    Subtraction,
    Multiplication,
    Division,
}

#[derive(Debug, Clone)]
pub enum TransformationOperation {
    Round,
    Ceil,
    Floor,

    Trim,
    TrimStart,
    TrimEnd,

    Lowercase,
    Uppercase,
}

#[derive(Debug, Clone)]
pub enum ValueOperand {
    Value(MedRecordValue),
    Evaluate(MedRecordAttribute),
    ArithmeticOperation(ArithmeticOperation, MedRecordAttribute, MedRecordValue),
    TransformationOperation(TransformationOperation, MedRecordAttribute),
    Slice(MedRecordAttribute, Range<usize>),
}

pub trait IntoValueOperand {
    fn into_value_operand(self) -> ValueOperand;
}

impl<T: Into<MedRecordValue>> IntoValueOperand for T {
    fn into_value_operand(self) -> ValueOperand {
        ValueOperand::Value(self.into())
    }
}
impl IntoValueOperand for NodeAttributeOperand {
    fn into_value_operand(self) -> ValueOperand {
        ValueOperand::Evaluate(self.into())
    }
}
impl IntoValueOperand for EdgeAttributeOperand {
    fn into_value_operand(self) -> ValueOperand {
        ValueOperand::Evaluate(self.into())
    }
}
impl IntoValueOperand for ValueOperand {
    fn into_value_operand(self) -> ValueOperand {
        self
    }
}

#[derive(Debug, Clone)]
pub struct NodeAttributeOperand(MedRecordAttribute);

impl From<NodeAttributeOperand> for MedRecordAttribute {
    fn from(val: NodeAttributeOperand) -> Self {
        val.0
    }
}

impl NodeAttributeOperand {
    pub fn greater(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gt(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn less(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Lt(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn greater_or_equal(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gte(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn less_or_equal(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Lte(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn equal(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Eq(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn not_equal(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Neq(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn r#in(self, operand: Vec<impl Into<MedRecordValue>>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::In(
            self.into(),
            operand.into_iter().map(|value| value.into()).collect(),
        ))
    }
    pub fn not_in(self, operand: Vec<impl Into<MedRecordValue>>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::NotIn(
            self.into(),
            operand.into_iter().map(|value| value.into()).collect(),
        ))
    }

    pub fn starts_with(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::StartsWith(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn ends_with(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::EndsWith(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn contains(self, operand: impl IntoValueOperand) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Contains(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn add(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(ArithmeticOperation::Addition, self.into(), value.into())
    }

    pub fn sub(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(
            ArithmeticOperation::Subtraction,
            self.into(),
            value.into(),
        )
    }

    pub fn mul(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(
            ArithmeticOperation::Multiplication,
            self.into(),
            value.into(),
        )
    }

    pub fn div(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(ArithmeticOperation::Division, self.into(), value.into())
    }

    pub fn round(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Round, self.into())
    }

    pub fn ceil(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Ceil, self.into())
    }

    pub fn floor(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Floor, self.into())
    }

    pub fn trim(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Trim, self.into())
    }

    pub fn trim_start(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::TrimStart, self.into())
    }

    pub fn trim_end(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::TrimEnd, self.into())
    }

    pub fn lowercase(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Lowercase, self.into())
    }

    pub fn uppercase(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Uppercase, self.into())
    }

    pub fn slice(self, range: Range<usize>) -> ValueOperand {
        ValueOperand::Slice(self.into(), range)
    }
}

#[derive(Debug, Clone)]
pub struct EdgeAttributeOperand(MedRecordAttribute);

impl From<EdgeAttributeOperand> for MedRecordAttribute {
    fn from(val: EdgeAttributeOperand) -> Self {
        val.0
    }
}

impl EdgeAttributeOperand {
    pub fn greater(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gt(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn less(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Lt(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn greater_or_equal(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gte(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn less_or_equal(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Lte(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn equal(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Eq(
            self.into(),
            operand.into_value_operand(),
        ))
    }
    pub fn not_equal(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Neq(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn r#in(self, operand: Vec<impl Into<MedRecordValue>>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::In(
            self.into(),
            operand.into_iter().map(|value| value.into()).collect(),
        ))
    }
    pub fn not_in(self, operand: Vec<impl Into<MedRecordValue>>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::NotIn(
            self.into(),
            operand.into_iter().map(|value| value.into()).collect(),
        ))
    }

    pub fn starts_with(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::StartsWith(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn ends_with(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::EndsWith(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn contains(self, operand: impl IntoValueOperand) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Contains(
            self.into(),
            operand.into_value_operand(),
        ))
    }

    pub fn add(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(ArithmeticOperation::Addition, self.into(), value.into())
    }

    pub fn sub(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(
            ArithmeticOperation::Subtraction,
            self.into(),
            value.into(),
        )
    }

    pub fn mul(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(
            ArithmeticOperation::Multiplication,
            self.into(),
            value.into(),
        )
    }

    pub fn div(self, value: impl Into<MedRecordValue>) -> ValueOperand {
        ValueOperand::ArithmeticOperation(ArithmeticOperation::Division, self.into(), value.into())
    }

    pub fn round(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Round, self.into())
    }

    pub fn ceil(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Ceil, self.into())
    }

    pub fn floor(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Floor, self.into())
    }

    pub fn trim(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Trim, self.into())
    }

    pub fn trim_start(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::TrimStart, self.into())
    }

    pub fn trim_end(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::TrimEnd, self.into())
    }

    pub fn lowercase(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Lowercase, self.into())
    }

    pub fn uppercase(self) -> ValueOperand {
        ValueOperand::TransformationOperation(TransformationOperation::Uppercase, self.into())
    }

    pub fn slice(self, range: Range<usize>) -> ValueOperand {
        ValueOperand::Slice(self.into(), range)
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndexInOperand {
    Vector(Vec<NodeIndex>),
    Operation(NodeOperation),
}

impl<T> From<Vec<T>> for NodeIndexInOperand
where
    T: Into<NodeIndex>,
{
    fn from(value: Vec<T>) -> NodeIndexInOperand {
        NodeIndexInOperand::Vector(value.into_iter().map(|value| value.into()).collect())
    }
}

impl From<NodeOperation> for NodeIndexInOperand {
    fn from(value: NodeOperation) -> Self {
        NodeIndexInOperand::Operation(value)
    }
}

pub(super) trait IntoVecNodeIndex {
    fn into_vec_node_index(self, medrecord: &MedRecord) -> Vec<NodeIndex>;
}

impl IntoVecNodeIndex for NodeIndexInOperand {
    fn into_vec_node_index(self, medrecord: &MedRecord) -> Vec<NodeIndex> {
        match self {
            NodeIndexInOperand::Vector(value) => value,
            NodeIndexInOperand::Operation(operation) => operation
                .evaluate(medrecord, medrecord.node_indices())
                .cloned()
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeIndexOperand;

impl NodeIndexOperand {
    pub fn greater(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gt(operand.into()))
    }
    pub fn less(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Lt(operand.into()))
    }
    pub fn greater_or_equal(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gte(operand.into()))
    }
    pub fn less_or_equal(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Lte(operand.into()))
    }

    pub fn equal(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Eq(operand.into()))
    }
    pub fn not_equal(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        self.equal(operand).not()
    }

    pub fn r#in(self, operand: impl Into<NodeIndexInOperand>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::In(Box::new(operand.into())))
    }
    pub fn not_in(self, operand: impl Into<NodeIndexInOperand>) -> NodeOperation {
        self.r#in(operand).not()
    }

    pub fn starts_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::StartsWith(operand.into()))
    }

    pub fn ends_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::EndsWith(operand.into()))
    }

    pub fn contains(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Contains(operand.into()))
    }
}

#[derive(Debug, Clone)]
pub struct NodeOperand;

impl NodeOperand {
    pub fn in_group(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::InGroup(operand.into())
    }

    pub fn has_attribute(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::HasAttribute(operand.into())
    }

    pub fn has_outgoing_edge_with(self, operation: EdgeOperation) -> NodeOperation {
        NodeOperation::HasOutgoingEdgeWith(operation.into())
    }
    pub fn has_incoming_edge_with(self, operation: EdgeOperation) -> NodeOperation {
        NodeOperation::HasIncomingEdgeWith(operation.into())
    }
    pub fn has_edge_with(self, operation: EdgeOperation) -> NodeOperation {
        NodeOperation::HasOutgoingEdgeWith(operation.clone().into())
            .or(NodeOperation::HasIncomingEdgeWith(operation.into()))
    }

    pub fn has_neighbor_with(self, operation: NodeOperation) -> NodeOperation {
        NodeOperation::HasNeighborWith(Box::new(operation))
    }

    pub fn attribute(self, attribute: impl Into<MedRecordAttribute>) -> NodeAttributeOperand {
        NodeAttributeOperand(attribute.into())
    }

    pub fn index(self) -> NodeIndexOperand {
        NodeIndexOperand
    }
}

pub fn node() -> NodeOperand {
    NodeOperand
}

#[derive(Debug, Clone)]
pub enum EdgeIndexInOperand {
    Vector(Vec<EdgeIndex>),
    Operation(EdgeOperation),
}

impl<T: Into<EdgeIndex>> From<Vec<T>> for EdgeIndexInOperand {
    fn from(value: Vec<T>) -> EdgeIndexInOperand {
        EdgeIndexInOperand::Vector(value.into_iter().map(|value| value.into()).collect())
    }
}

impl From<EdgeOperation> for EdgeIndexInOperand {
    fn from(value: EdgeOperation) -> Self {
        EdgeIndexInOperand::Operation(value)
    }
}

pub(super) trait IntoVecEdgeIndex {
    fn into_vec_edge_index(self, medrecord: &MedRecord) -> Vec<EdgeIndex>;
}

impl IntoVecEdgeIndex for EdgeIndexInOperand {
    fn into_vec_edge_index(self, medrecord: &MedRecord) -> Vec<EdgeIndex> {
        match self {
            EdgeIndexInOperand::Vector(value) => value,
            EdgeIndexInOperand::Operation(operation) => operation
                .evaluate(medrecord, medrecord.edge_indices())
                .copied()
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdgeIndexOperand;

impl EdgeIndexOperand {
    pub fn greater(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gt(operand))
    }
    pub fn less(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Lt(operand))
    }
    pub fn greater_or_equal(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gte(operand))
    }
    pub fn less_or_equal(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Lte(operand))
    }

    pub fn equal(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Eq(operand))
    }
    pub fn not_equal(self, operand: EdgeIndex) -> EdgeOperation {
        self.equal(operand).not()
    }

    pub fn r#in(self, operand: impl Into<EdgeIndexInOperand>) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::In(Box::new(operand.into())))
    }
    pub fn not_in(self, operand: impl Into<EdgeIndexInOperand>) -> EdgeOperation {
        self.r#in(operand).not()
    }
}

#[derive(Debug, Clone)]
pub struct EdgeOperand;

impl EdgeOperand {
    pub fn connected_target(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedSource(operand.into())
    }

    pub fn connected_source(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedTarget(operand.into())
    }

    pub fn connected(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        let attribute = operand.into();

        EdgeOperation::ConnectedSource(attribute.clone())
            .or(EdgeOperation::ConnectedTarget(attribute))
    }

    pub fn has_attribute(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::HasAttribute(operand.into())
    }

    pub fn connected_source_with(self, operation: NodeOperation) -> EdgeOperation {
        EdgeOperation::ConnectedSourceWith(operation.into())
    }

    pub fn connected_target_with(self, operation: NodeOperation) -> EdgeOperation {
        EdgeOperation::ConnectedTargetWith(operation.into())
    }

    pub fn connected_with(self, operation: NodeOperation) -> EdgeOperation {
        EdgeOperation::ConnectedSourceWith(operation.clone().into())
            .or(EdgeOperation::ConnectedTargetWith(operation.into()))
    }

    pub fn has_parallel_edges_with(self, operation: EdgeOperation) -> EdgeOperation {
        EdgeOperation::HasParallelEdgesWith(Box::new(operation))
    }

    pub fn has_parallel_edges_with_self_comparison(
        self,
        operation: EdgeOperation,
    ) -> EdgeOperation {
        EdgeOperation::HasParallelEdgesWithSelfComparison(Box::new(operation))
    }

    pub fn attribute(self, attribute: impl Into<MedRecordAttribute>) -> EdgeAttributeOperand {
        EdgeAttributeOperand(attribute.into())
    }

    pub fn index(self) -> EdgeIndexOperand {
        EdgeIndexOperand
    }
}

pub fn edge() -> EdgeOperand {
    EdgeOperand
}

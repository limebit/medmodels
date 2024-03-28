use super::{
    edge_operation::EdgeIndexOperation,
    node_operation::{NodeIndexOperation, NodeOperation},
    AttributeOperation, EdgeOperation,
};
use crate::medrecord::{EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex};

pub(super) type ValueOperand = MedRecordValue;

#[derive(Debug, Clone)]
pub struct NodeAttributeOperand(MedRecordAttribute);

impl From<NodeAttributeOperand> for MedRecordAttribute {
    fn from(val: NodeAttributeOperand) -> Self {
        val.0
    }
}

impl NodeAttributeOperand {
    pub fn greater_than(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gt(self, operand.into()))
    }
    pub fn less_than_or_equal_to(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gt(self, operand.into())).not()
    }

    pub fn greather_than_or_equal_to(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gte(self, operand.into()))
    }
    pub fn less_than(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Gte(self, operand.into())).not()
    }

    pub fn equal_to(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Eq(self, operand.into()))
    }
    pub fn not_equal_to(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Eq(self, operand.into())).not()
    }

    pub fn r#in(self, operand: impl Into<Vec<ValueOperand>>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::In(self, operand.into()))
    }
    pub fn not_in(self, operand: impl Into<Vec<ValueOperand>>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::In(self, operand.into())).not()
    }

    pub fn starts_with(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::StartsWith(self, operand.into()))
    }
    pub fn not_starts_with(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::StartsWith(self, operand.into())).not()
    }

    pub fn ends_with(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::EndsWith(self, operand.into()))
    }
    pub fn not_ends_with(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::EndsWith(self, operand.into())).not()
    }

    pub fn contains(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Contains(self, operand.into()))
    }
    pub fn not_contains(self, operand: impl Into<ValueOperand>) -> NodeOperation {
        NodeOperation::Attribute(AttributeOperation::Contains(self, operand.into())).not()
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
    pub fn greater_than(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gt(self, operand.into()))
    }
    pub fn less_than_or_equal_to(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gt(self, operand.into())).not()
    }

    pub fn greather_than_or_equal_to(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gte(self, operand.into()))
    }
    pub fn less_than(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Gte(self, operand.into())).not()
    }

    pub fn equal_to(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Eq(self, operand.into()))
    }
    pub fn not_equal_to(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Eq(self, operand.into())).not()
    }

    pub fn r#in(self, operand: impl Into<Vec<ValueOperand>>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::In(self, operand.into()))
    }
    pub fn not_in(self, operand: impl Into<Vec<ValueOperand>>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::In(self, operand.into())).not()
    }

    pub fn starts_with(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::StartsWith(self, operand.into()))
    }
    pub fn not_starts_with(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::StartsWith(self, operand.into())).not()
    }

    pub fn ends_with(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::EndsWith(self, operand.into()))
    }
    pub fn not_ends_with(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::EndsWith(self, operand.into())).not()
    }

    pub fn contains(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Contains(self, operand.into()))
    }
    pub fn not_contains(self, operand: impl Into<ValueOperand>) -> EdgeOperation {
        EdgeOperation::Attribute(AttributeOperation::Contains(self, operand.into())).not()
    }
}
pub struct NodeIndexOperand;

impl NodeIndexOperand {
    pub fn greater_than(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gt(operand.into()))
    }
    pub fn less_than_or_equal_to(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gt(operand.into())).not()
    }

    pub fn greather_than_or_equal_to(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gte(operand.into()))
    }
    pub fn less_than(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Gte(operand.into())).not()
    }

    pub fn equal_to(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Eq(operand.into()))
    }
    pub fn not_equal_to(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Eq(operand.into())).not()
    }

    pub fn r#in(self, operand: impl Into<Vec<NodeIndex>>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::In(operand.into()))
    }
    pub fn not_in(self, operand: impl Into<Vec<NodeIndex>>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::In(operand.into())).not()
    }

    pub fn starts_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::StartsWith(operand.into()))
    }
    pub fn not_starts_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::StartsWith(operand.into())).not()
    }

    pub fn ends_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::EndsWith(operand.into()))
    }
    pub fn not_ends_with(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::EndsWith(operand.into())).not()
    }

    pub fn contains(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Contains(operand.into()))
    }
    pub fn not_contains(self, operand: impl Into<NodeIndex>) -> NodeOperation {
        NodeOperation::Index(NodeIndexOperation::Contains(operand.into())).not()
    }
}

pub struct NodeOperand;

impl NodeOperand {
    pub fn connected(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::Connected(self.attribute(operand))
    }
    pub fn not_connected(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::Connected(self.attribute(operand)).not()
    }

    pub fn connected_with(
        self,
        operand: impl Into<MedRecordAttribute>,
        operation: EdgeOperation,
    ) -> NodeOperation {
        NodeOperation::ConnectedWith(self.attribute(operand), Box::new(operation))
    }
    pub fn not_connected_with(
        self,
        operand: impl Into<MedRecordAttribute>,
        operation: EdgeOperation,
    ) -> NodeOperation {
        NodeOperation::ConnectedWith(self.attribute(operand), Box::new(operation)).not()
    }

    pub fn in_group(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::InGroup(self.attribute(operand))
    }
    pub fn not_in_group(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::InGroup(self.attribute(operand)).not()
    }

    pub fn has_attribute(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::HasAttribute(self.attribute(operand))
    }
    pub fn not_has_attribute(self, operand: impl Into<MedRecordAttribute>) -> NodeOperation {
        NodeOperation::HasAttribute(self.attribute(operand)).not()
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

pub struct EdgeIndexOperand;

impl EdgeIndexOperand {
    pub fn greater_than(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gt(operand))
    }
    pub fn less_than_or_equal_to(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gt(operand)).not()
    }

    pub fn greather_than_or_equal_to(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gte(operand))
    }
    pub fn less_than(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Gte(operand)).not()
    }

    pub fn equal_to(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Eq(operand))
    }
    pub fn not_equal_to(self, operand: EdgeIndex) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::Eq(operand)).not()
    }

    pub fn r#in(self, operand: Vec<EdgeIndex>) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::In(operand))
    }
    pub fn not_in(self, operand: Vec<EdgeIndex>) -> EdgeOperation {
        EdgeOperation::Index(EdgeIndexOperation::In(operand)).not()
    }
}

pub struct EdgeOperand;

impl EdgeOperand {
    pub fn connected_target(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedSource(self.attribute(operand))
    }
    pub fn not_connected_target(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedSource(self.attribute(operand)).not()
    }

    pub fn connected_source(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedTarget(self.attribute(operand))
    }
    pub fn not_connected_source(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::ConnectedTarget(self.attribute(operand)).not()
    }

    pub fn has_attribute(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::HasAttribute(self.attribute(operand))
    }
    pub fn not_has_attribute(self, operand: impl Into<MedRecordAttribute>) -> EdgeOperation {
        EdgeOperation::HasAttribute(self.attribute(operand)).not()
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

use super::{inferred::InferredGroupSchema, AttributeSchema, Attributes, EdgeIndex, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, Group, MedRecordAttribute},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProvidedGroupSchema {
    pub nodes: AttributeSchema,
    pub edges: AttributeSchema,
}

enum AttributeSchemaKind<'a> {
    Node(&'a NodeIndex),
    Edge(&'a EdgeIndex),
}

impl AttributeSchemaKind<'_> {
    fn error_message(&self, key: &MedRecordAttribute, data_type: &DataType) -> String {
        match self {
            Self::Node(index) => format!(
                "Attribute {} of type {} not found on node with index {}",
                key, data_type, index
            ),
            Self::Edge(index) => format!(
                "Attribute {} of type {} not found on edge with index {}",
                key, data_type, index
            ),
        }
    }

    fn error_message_expected(
        &self,
        key: &MedRecordAttribute,
        data_type: &DataType,
        expected_data_type: &DataType,
    ) -> String {
        match self {
            Self::Node(index) => format!(
                "Attribute {} of node with index {} is of type {}. Expected {}.",
                key, index, data_type, expected_data_type
            ),
            Self::Edge(index) => format!(
                "Attribute {} of node with index {} is of type {}. Expected {}.",
                key, index, data_type, expected_data_type
            ),
        }
    }

    fn error_message_too_many(&self, attributes: Vec<String>) -> String {
        match self {
            Self::Node(index) => format!(
                "Attributes [{}] of node with index {} do not exist in schema.",
                attributes.join(", "),
                index
            ),
            Self::Edge(index) => format!(
                "Attributes [{}] of edge with index {} do not exist in schema.",
                attributes.join(", "),
                index
            ),
        }
    }
}

impl From<InferredGroupSchema> for ProvidedGroupSchema {
    fn from(value: InferredGroupSchema) -> Self {
        Self {
            nodes: value.nodes,
            edges: value.edges,
        }
    }
}

impl ProvidedGroupSchema {
    fn validate_attribute_schema(
        attributes: &Attributes,
        attribute_schema: &AttributeSchema,
        kind: AttributeSchemaKind,
    ) -> Result<(), GraphError> {
        for (key, schema) in attribute_schema {
            let value = attributes.get(key).ok_or(GraphError::SchemaError(
                kind.error_message(key, &schema.data_type),
            ))?;

            let data_type = DataType::from(value);

            if !schema.data_type.evaluate(&data_type) {
                return Err(GraphError::SchemaError(kind.error_message_expected(
                    key,
                    &data_type,
                    &schema.data_type,
                )));
            }
        }

        let attributes_not_in_schema = attributes
            .keys()
            .filter(|attribute| !attribute_schema.contains_key(*attribute))
            .map(|attribute| attribute.to_string())
            .collect::<Vec<_>>();

        match attributes_not_in_schema.len() {
            0 => (),
            _ => {
                return Err(GraphError::SchemaError(
                    kind.error_message_too_many(attributes_not_in_schema),
                ));
            }
        }

        Ok(())
    }

    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        Self::validate_attribute_schema(attributes, &self.nodes, AttributeSchemaKind::Node(index))
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        Self::validate_attribute_schema(attributes, &self.edges, AttributeSchemaKind::Edge(index))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProvidedSchema {
    pub groups: HashMap<Group, ProvidedGroupSchema>,
    pub default: ProvidedGroupSchema,
}

impl ProvidedSchema {
    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        let group_schema = group.and_then(|group| self.groups.get(group));

        match group_schema {
            Some(group_schema) => group_schema.validate_node(index, attributes),
            None => self.default.validate_node(index, attributes),
        }
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        let group_schema = group.and_then(|group| self.groups.get(group));

        match group_schema {
            Some(group_schema) => group_schema.validate_edge(index, attributes),
            None => self.default.validate_edge(index, attributes),
        }
    }
}

#[cfg(test)]
mod test {
    use super::{ProvidedGroupSchema, ProvidedSchema};
    use crate::{
        errors::GraphError,
        medrecord::{Attributes, DataType, EdgeIndex, NodeIndex},
    };
    use std::collections::HashMap;

    #[test]
    fn test_validate_node_default_schema() {
        let strict_schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_invalid_validate_node_default_schema() {
        let schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_node_group_schema() {
        let strict_schema = ProvidedSchema {
            groups: HashMap::from([(
                "group".into(),
                ProvidedGroupSchema {
                    nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    edges: Default::default(),
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        // Checking schema of non existing group should fail because no default schema exists
        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"test".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_edge_default_schema() {
        let strict_schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_invalid_validate_edge_default_schema() {
        let schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_edge_group_schema() {
        let strict_schema = ProvidedSchema {
            groups: HashMap::from([(
                "group".into(),
                ProvidedGroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
            },
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        // Checking schema of non existing group should fail because no default schema exists
        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"test".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }
}

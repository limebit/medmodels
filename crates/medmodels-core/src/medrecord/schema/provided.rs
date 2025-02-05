use super::{inferred::InferredGroupSchema, AttributeSchema, Attributes, EdgeIndex, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, Group},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProvidedGroupSchema {
    pub nodes: AttributeSchema,
    pub edges: AttributeSchema,
    pub strict: bool,
}

impl From<InferredGroupSchema> for ProvidedGroupSchema {
    fn from(value: InferredGroupSchema) -> Self {
        Self {
            nodes: value.nodes,
            edges: value.edges,
            strict: false,
        }
    }
}

impl ProvidedGroupSchema {
    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        for (key, schema) in &self.nodes {
            let value = attributes.get(key).ok_or(GraphError::SchemaError(format!(
                "Attribute {} of type {} not found on node with index {}",
                key, schema.data_type, index
            )))?;

            let data_type = DataType::from(value);

            if !schema.data_type.evaluate(&data_type) {
                return Err(GraphError::SchemaError(format!(
                    "Attribute {} of node with index {} is of type {}. Expected {}.",
                    key, index, data_type, schema.data_type
                )));
            }
        }

        if self.strict {
            let attributes = attributes.keys().collect::<HashSet<_>>();
            let schema_attributes = self.nodes.keys().collect::<HashSet<_>>();
            let attributes_not_in_schema = attributes
                .difference(&schema_attributes)
                .map(|attribute| attribute.to_string())
                .collect::<Vec<_>>();

            match attributes_not_in_schema.len() {
                0 => (),
                1 => {
                    let attribute_not_in_schema = attributes_not_in_schema
                        .first()
                        .expect("Attribute must exist.");

                    return Err(GraphError::SchemaError(format!(
                        "Attribute {} of node with index {} does not exist in strict schema.",
                        attribute_not_in_schema, index
                    )));
                }
                _ => {
                    return Err(GraphError::SchemaError(format!(
                        "Attributes {} of node with index {} do not exist in strict schema.",
                        attributes_not_in_schema.join(", "),
                        index
                    )));
                }
            }
        }

        Ok(())
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        for (key, schema) in &self.edges {
            let value = attributes.get(key).ok_or(GraphError::SchemaError(format!(
                "Attribute {} of type {} not found on edge with index {}",
                key, schema.data_type, index
            )))?;

            let data_type = DataType::from(value);

            if !schema.data_type.evaluate(&data_type) {
                return Err(GraphError::SchemaError(format!(
                    "Attribute {} of edge with index {} is of type {}. Expected {}.",
                    key, index, data_type, schema.data_type
                )));
            }
        }

        if self.strict {
            let attributes = attributes.keys().collect::<HashSet<_>>();
            let schema_attributes = self.edges.keys().collect::<HashSet<_>>();
            let attributes_not_in_schema = attributes
                .difference(&schema_attributes)
                .map(|attribute| attribute.to_string())
                .collect::<Vec<_>>();

            match attributes_not_in_schema.len() {
                0 => (),
                1 => {
                    let attribute_not_in_schema = attributes_not_in_schema
                        .first()
                        .expect("Attribute must exist.");

                    return Err(GraphError::SchemaError(format!(
                        "Attribute {} of edge with index {} does not exist in strict schema.",
                        attribute_not_in_schema, index
                    )));
                }
                _ => {
                    return Err(GraphError::SchemaError(format!(
                        "Attributes {} of edge with index {} do not exist in strict schema.",
                        attributes_not_in_schema.join(", "),
                        index
                    )));
                }
            }
        }

        Ok(())
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
        medrecord::{AttributeType, Attributes, DataType, EdgeIndex, NodeIndex},
    };
    use std::collections::HashMap;

    #[test]
    fn test_validate_node_default_schema() {
        let strict_schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: HashMap::from([(
                    "attribute".into(),
                    (DataType::Int, AttributeType::Unstructured).into(),
                )]),
                edges: Default::default(),
                strict: true,
            },
        };
        let non_strict_schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: HashMap::from([(
                    "attribute".into(),
                    (DataType::Int, AttributeType::Unstructured).into(),
                )]),
                edges: Default::default(),
                strict: false,
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
        assert!(non_strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());
    }

    #[test]
    fn test_invalid_validate_node_default_schema() {
        let schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
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
                    nodes: HashMap::from([(
                        "attribute".into(),
                        (DataType::Int, AttributeType::Unstructured).into(),
                    )]),
                    edges: Default::default(),
                    strict: true,
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
            },
        };
        let non_strict_schema = ProvidedSchema {
            groups: HashMap::from([(
                "group".into(),
                ProvidedGroupSchema {
                    nodes: HashMap::from([(
                        "attribute".into(),
                        (DataType::Int, AttributeType::Unstructured).into(),
                    )]),
                    edges: Default::default(),
                    strict: false,
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
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
        assert!(non_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());

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
                edges: HashMap::from([(
                    "attribute".into(),
                    (DataType::Int, AttributeType::Unstructured).into(),
                )]),
                strict: true,
            },
        };
        let non_strict_schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([(
                    "attribute".into(),
                    (DataType::Int, AttributeType::Unstructured).into(),
                )]),
                strict: false,
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
        assert!(non_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());
    }

    #[test]
    fn test_invalid_validate_edge_default_schema() {
        let schema = ProvidedSchema {
            groups: Default::default(),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
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
                    edges: HashMap::from([(
                        "attribute".into(),
                        (DataType::Int, AttributeType::Unstructured).into(),
                    )]),
                    strict: true,
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
            },
        };
        let non_strict_schema = ProvidedSchema {
            groups: HashMap::from([(
                "group".into(),
                ProvidedGroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([(
                        "attribute".into(),
                        (DataType::Int, AttributeType::Unstructured).into(),
                    )]),
                    strict: false,
                },
            )]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
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
        assert!(non_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());

        // Checking schema of non existing group should fail because no default schema exists
        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"test".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }
}

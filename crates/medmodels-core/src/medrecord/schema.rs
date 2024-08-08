#![allow(dead_code)]

use super::{Attributes, EdgeIndex, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, Group, MedRecordAttribute},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttributeType {
    Categorical,
    Continuous,
    Temporal,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttributeDataType {
    pub data_type: DataType,
    pub attribute_type: Option<AttributeType>,
}

impl AttributeDataType {
    pub fn new(data_type: DataType, attribute_type: Option<AttributeType>) -> Self {
        Self {
            data_type,
            attribute_type,
        }
    }
}

impl From<DataType> for AttributeDataType {
    fn from(value: DataType) -> Self {
        Self {
            data_type: value,
            attribute_type: None,
        }
    }
}

impl From<(DataType, AttributeType)> for AttributeDataType {
    fn from(value: (DataType, AttributeType)) -> Self {
        Self {
            data_type: value.0,
            attribute_type: Some(value.1),
        }
    }
}

type AttributeSchema = HashMap<MedRecordAttribute, AttributeDataType>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroupSchema {
    pub nodes: AttributeSchema,
    pub edges: AttributeSchema,
    pub strict: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Schema {
    pub groups: HashMap<Group, GroupSchema>,
    pub default: Option<GroupSchema>,
    pub strict: Option<bool>,
}

impl GroupSchema {
    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
        strict: bool,
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

        if self.strict.unwrap_or(strict) {
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
        strict: bool,
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

        if self.strict.unwrap_or(strict) {
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

impl Schema {
    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        let group_schema = group.and_then(|group| self.groups.get(group));

        match (group_schema, &self.default, self.strict) {
            (Some(group_schema), _, Some(true)) => {
                group_schema.validate_node(index, attributes, true)?;

                Ok(())
            }
            (Some(group_schema), _, _) => {
                group_schema.validate_node(index, attributes, false)?;

                Ok(())
            }
            (_, Some(defalt_schema), Some(true)) => {
                defalt_schema.validate_node(index, attributes, true)
            }
            (_, Some(default_schema), _) => default_schema.validate_node(index, attributes, false),
            (None, None, None) | (None, None, Some(false)) => Ok(()),

            _ => Err(GraphError::SchemaError(format!(
                "No schema provided for node {} wit no group",
                index
            ))),
        }
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        let group_schema = group.and_then(|group| self.groups.get(group));

        match (group_schema, &self.default, self.strict) {
            (Some(group_schema), _, Some(true)) => {
                group_schema.validate_edge(index, attributes, true)?;

                Ok(())
            }
            (Some(group_schema), _, _) => {
                group_schema.validate_edge(index, attributes, false)?;

                Ok(())
            }
            (_, Some(defalt_schema), Some(true)) => {
                defalt_schema.validate_edge(index, attributes, true)
            }
            (_, Some(default_schema), _) => default_schema.validate_edge(index, attributes, false),
            (None, None, None) | (None, None, Some(false)) => Ok(()),

            _ => Err(GraphError::SchemaError(format!(
                "No schema provided for edge {} wit no group",
                index
            ))),
        }
    }
}

impl Default for Schema {
    fn default() -> Self {
        Schema {
            groups: HashMap::new(),
            default: None,
            strict: Some(false),
        }
    }
}

#[cfg(test)]
mod test {
    use super::{GroupSchema, Schema};
    use crate::{
        errors::GraphError,
        medrecord::{Attributes, DataType, EdgeIndex, NodeIndex},
    };
    use std::collections::HashMap;

    #[test]
    fn test_validate_node_default_schema() {
        let strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: None,
            }),
            strict: Some(true),
        };
        let second_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: Some(true),
            }),
            strict: Some(false),
        };
        let non_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: None,
            }),
            strict: Some(false),
        };
        let second_non_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: Some(false),
            }),
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());
        assert!(second_strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(non_strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());
        assert!(second_non_strict_schema
            .validate_node(&index, &attributes, None)
            .is_ok());
    }

    #[test]
    fn test_invalid_validate_node_default_schema() {
        let schema = Schema {
            groups: Default::default(),
            default: None,
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(schema
            .validate_node(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_node_group_schema() {
        let strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    edges: Default::default(),
                    strict: None,
                },
            )]),
            default: None,
            strict: Some(true),
        };
        let second_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    edges: Default::default(),
                    strict: Some(true),
                },
            )]),
            default: None,
            strict: Some(false),
        };
        let non_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    edges: Default::default(),
                    strict: None,
                },
            )]),
            default: None,
            strict: Some(false),
        };
        let second_non_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    edges: Default::default(),
                    strict: Some(false),
                },
            )]),
            default: None,
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: NodeIndex = 0.into();

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());
        assert!(second_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(non_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());
        assert!(second_non_strict_schema
            .validate_node(&index, &attributes, Some(&"group".into()))
            .is_ok());

        // Checking schema of non existing group should fail because no default schema exists
        assert!(strict_schema
            .validate_node(&index, &attributes, Some(&"test".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_edge_default_schema() {
        let strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: None,
            }),
            strict: Some(true),
        };
        let second_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: Some(true),
            }),
            strict: Some(false),
        };
        let non_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: None,
            }),
            strict: Some(false),
        };
        let second_non_strict_schema = Schema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: Some(false),
            }),
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(non_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());
        assert!(second_non_strict_schema
            .validate_edge(&index, &attributes, None)
            .is_ok());
    }

    #[test]
    fn test_invalid_validate_edge_default_schema() {
        let schema = Schema {
            groups: Default::default(),
            default: None,
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(schema
            .validate_edge(&index, &attributes, None)
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }

    #[test]
    fn test_validate_edge_group_schema() {
        let strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    strict: None,
                },
            )]),
            default: None,
            strict: Some(true),
        };
        let second_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    strict: Some(true),
                },
            )]),
            default: None,
            strict: Some(false),
        };
        let non_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    strict: None,
                },
            )]),
            default: None,
            strict: Some(false),
        };
        let second_non_strict_schema = Schema {
            groups: HashMap::from([(
                "group".into(),
                GroupSchema {
                    nodes: Default::default(),
                    edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                    strict: Some(false),
                },
            )]),
            default: None,
            strict: Some(true),
        };

        let attributes: Attributes = HashMap::from([("attribute".into(), 1.into())]);
        let index: EdgeIndex = 0;

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());

        let attributes: Attributes = HashMap::from([("attribute".into(), "1".into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));

        let attributes: Attributes =
            HashMap::from([("attribute".into(), 1.into()), ("extra".into(), 1.into())]);

        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(second_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
        assert!(non_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());
        assert!(second_non_strict_schema
            .validate_edge(&index, &attributes, Some(&"group".into()))
            .is_ok());

        // Checking schema of non existing group should fail because no default schema exists
        assert!(strict_schema
            .validate_edge(&index, &attributes, Some(&"test".into()))
            .is_err_and(|e| matches!(e, GraphError::SchemaError(_))));
    }
}

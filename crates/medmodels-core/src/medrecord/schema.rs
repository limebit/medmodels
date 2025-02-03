use super::{Attributes, EdgeIndex, MedRecord, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, Group, MedRecordAttribute},
};
use medmodels_utils::aliases::MrHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub enum Schema {
    Inferred(InferredSchema),
    Provided(),
}

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

impl GroupSchema {
    pub(crate) fn infer(nodes: Vec<&Attributes>, edges: Vec<&Attributes>) -> Self {
        Self {
            nodes: Self::infer_attribute_schema(nodes),
            edges: Self::infer_attribute_schema(edges),
            strict: None, // TODO: Infer strictness
        }
    }

    fn infer_attribute_schema(attributes: Vec<&Attributes>) -> AttributeSchema {
        let mut schema = AttributeSchema::new();

        for attribute in attributes {
            for (key, value) in attribute {
                let data_type = DataType::from(value);

                match schema.entry(key.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        let existing_data_type = entry.into_mut();

                        if !existing_data_type.data_type.evaluate(&data_type) {
                            *existing_data_type = AttributeDataType::new(
                                DataType::Union((
                                    Box::new(existing_data_type.data_type.clone()),
                                    Box::new(data_type),
                                )),
                                None, // TODO
                            );
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(AttributeDataType::new(data_type.clone(), None));
                    }
                }
            }
        }

        schema
    }

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InferredSchema {
    pub groups: HashMap<Group, GroupSchema>,
    pub default: Option<GroupSchema>,
    pub strict: Option<bool>,
}

impl InferredSchema {
    pub(crate) fn infer(medrecord: &MedRecord) -> InferredSchema {
        let mut group_mapping = medrecord
            .groups()
            .map(|group| (group, (Vec::new(), Vec::new())))
            .collect::<MrHashMap<_, _>>();

        let mut default_group = (Vec::new(), Vec::new());

        for node_index in medrecord.node_indices() {
            let mut groups_of_node = medrecord
                .groups_of_node(node_index)
                .expect("Node must exist.")
                .peekable();

            if groups_of_node.peek().is_none() {
                default_group.0.push(node_index);
            }

            for group in groups_of_node {
                let group_nodes = &mut group_mapping.get_mut(&group).expect("Group must exist.").0;

                group_nodes.push(node_index);
            }
        }

        for edge_index in medrecord.edge_indices() {
            let mut groups_of_edge = medrecord
                .groups_of_edge(edge_index)
                .expect("Edge must exist.")
                .peekable();

            if groups_of_edge.peek().is_none() {
                default_group.1.push(edge_index);
            }

            for group in groups_of_edge {
                let group_edges = &mut group_mapping.get_mut(&group).expect("Group must exist.").1;

                group_edges.push(edge_index);
            }
        }

        let group_schemas =
            group_mapping
                .into_iter()
                .map(|(group, (nodes_in_group, edges_in_group))| {
                    let node_attributes = nodes_in_group
                        .into_iter()
                        .map(|node| medrecord.node_attributes(node).expect("Node must exist."))
                        .collect::<Vec<_>>();
                    let edge_attributes = edges_in_group
                        .into_iter()
                        .map(|edge| medrecord.edge_attributes(edge).expect("Edge must exist."))
                        .collect::<Vec<_>>();

                    let schema = GroupSchema::infer(node_attributes, edge_attributes);

                    (group.clone(), schema)
                });

        let default_schema = GroupSchema::infer(
            default_group
                .0
                .into_iter()
                .map(|node| medrecord.node_attributes(node).expect("Node must exist."))
                .collect::<Vec<_>>(),
            default_group
                .1
                .into_iter()
                .map(|edge| medrecord.edge_attributes(edge).expect("Edge must exist."))
                .collect::<Vec<_>>(),
        );

        Self {
            groups: group_schemas.collect(),
            default: Some(default_schema),
            strict: None,
        }
    }

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

impl Default for InferredSchema {
    fn default() -> Self {
        InferredSchema {
            groups: HashMap::new(),
            default: None,
            strict: Some(false),
        }
    }
}

#[cfg(test)]
mod test {
    use super::{GroupSchema, InferredSchema};
    use crate::{
        errors::GraphError,
        medrecord::{Attributes, DataType, EdgeIndex, NodeIndex},
    };
    use std::collections::HashMap;

    #[test]
    fn test_validate_node_default_schema() {
        let strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: None,
            }),
            strict: Some(true),
        };
        let second_strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: Some(true),
            }),
            strict: Some(false),
        };
        let non_strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: HashMap::from([("attribute".into(), DataType::Int.into())]),
                edges: Default::default(),
                strict: None,
            }),
            strict: Some(false),
        };
        let second_non_strict_schema = InferredSchema {
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
        let schema = InferredSchema {
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
        let strict_schema = InferredSchema {
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
        let second_strict_schema = InferredSchema {
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
        let non_strict_schema = InferredSchema {
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
        let second_non_strict_schema = InferredSchema {
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
        let strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: None,
            }),
            strict: Some(true),
        };
        let second_strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: Some(true),
            }),
            strict: Some(false),
        };
        let non_strict_schema = InferredSchema {
            groups: Default::default(),
            default: Some(GroupSchema {
                nodes: Default::default(),
                edges: HashMap::from([("attribute".into(), DataType::Int.into())]),
                strict: None,
            }),
            strict: Some(false),
        };
        let second_non_strict_schema = InferredSchema {
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
        let schema = InferredSchema {
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
        let strict_schema = InferredSchema {
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
        let second_strict_schema = InferredSchema {
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
        let non_strict_schema = InferredSchema {
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
        let second_non_strict_schema = InferredSchema {
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

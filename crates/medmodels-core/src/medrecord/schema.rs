#![allow(dead_code)]

use super::{Attributes, EdgeIndex, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, Group, MedRecordAttribute},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

type AttributeSchema = HashMap<MedRecordAttribute, DataType>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GroupSchema {
    pub nodes: AttributeSchema,
    pub edges: AttributeSchema,
    pub strict: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
                key, schema, index
            )))?;

            let data_type = DataType::from(value);

            if !schema.evaluate(&data_type) {
                return Err(GraphError::SchemaError(format!(
                    "Attribute {} of node with index {} is of type {}. Expected {}.",
                    key, index, data_type, schema
                )));
            }
        }

        if strict {
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
                key, schema, index
            )))?;

            let data_type = DataType::from(value);

            if !schema.evaluate(&data_type) {
                return Err(GraphError::SchemaError(format!(
                    "Attribute {} of edge with index {} is of type {}. Expected {}.",
                    key, index, data_type, schema
                )));
            }
        }

        if strict {
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
                group_schema.validate_node(index, attributes, false)?;

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
            (_, None, None) | (_, None, Some(false)) => Ok(()),

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
                group_schema.validate_edge(index, attributes, false)?;

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
            (_, None, None) | (_, None, Some(false)) => Ok(()),

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

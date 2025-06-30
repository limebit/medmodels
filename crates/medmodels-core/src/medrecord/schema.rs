use super::{Attributes, EdgeIndex, Group, MedRecord, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, MedRecordAttribute},
};
use medmodels_utils::aliases::MrHashMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::Entry, HashMap},
    ops::Deref,
};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum AttributeType {
    Categorical,
    Continuous,
    Temporal,
    Unstructured,
}

impl AttributeType {
    pub fn infer(data_type: &DataType) -> Self {
        match data_type {
            DataType::String => Self::Unstructured,
            DataType::Int => Self::Continuous,
            DataType::Float => Self::Continuous,
            DataType::Bool => Self::Categorical,
            DataType::DateTime => Self::Temporal,
            DataType::Duration => Self::Temporal,
            DataType::Null => Self::Unstructured,
            DataType::Any => Self::Unstructured,
            DataType::Union((first_dataype, second_dataype)) => {
                Self::infer(first_dataype).merge(&Self::infer(second_dataype))
            }
            DataType::Option(dataype) => Self::infer(dataype),
        }
    }

    fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Categorical, Self::Unstructured) | (Self::Unstructured, Self::Categorical) => {
                Self::Unstructured
            }
            (Self::Categorical, _) | (_, Self::Categorical) => Self::Categorical,
            (Self::Continuous, Self::Continuous) => Self::Continuous,
            (Self::Temporal, Self::Temporal) => Self::Temporal,
            _ => Self::Unstructured,
        }
    }
}

impl DataType {
    fn merge(&self, other: &Self) -> Self {
        if self.evaluate(other) {
            self.clone()
        } else {
            match (self, other) {
                (Self::Null, _) => Self::Option(Box::new(other.clone())),
                (_, Self::Null) => Self::Option(Box::new(self.clone())),
                (_, Self::Any) => Self::Any,
                (Self::Option(option1), Self::Option(option2)) => {
                    Self::Option(Box::new(option1.merge(option2)))
                }
                (Self::Option(option), _) => Self::Option(Box::new(option.merge(other))),
                (_, Self::Option(option)) => Self::Option(Box::new(self.merge(option))),
                _ => Self::Union((Box::new(self.clone()), Box::new(other.clone()))),
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttributeDataType {
    data_type: DataType,
    attribute_type: AttributeType,
}

impl AttributeDataType {
    fn validate(data_type: &DataType, attribute_type: &AttributeType) -> Result<(), GraphError> {
        match (attribute_type, data_type) {
            (AttributeType::Categorical, _) => Ok(()),
            (AttributeType::Unstructured, _) => Ok(()),

            (_, DataType::Option(option)) => Self::validate(option, attribute_type),
            (_, DataType::Union((first_datatype, second_datatype))) => {
                Self::validate(first_datatype, attribute_type)?;
                Self::validate(second_datatype, attribute_type)
            }

            (AttributeType::Continuous, DataType::Int | DataType::Float | DataType::Null) => Ok(()),
            (AttributeType::Continuous, _) => Err(GraphError::SchemaError(
                "Continuous attribute must be of (sub-)type Int or Float.".to_string(),
            )),

            (AttributeType::Temporal, DataType::DateTime | DataType::Duration | DataType::Null) => {
                Ok(())
            }
            (AttributeType::Temporal, _) => Err(GraphError::SchemaError(
                "Temporal attribute must be of (sub-)type DateTime or Duration.".to_string(),
            )),
        }
    }

    pub fn new(data_type: DataType, attribute_type: AttributeType) -> Result<Self, GraphError> {
        Self::validate(&data_type, &attribute_type)?;

        Ok(Self {
            data_type,
            attribute_type,
        })
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn attribute_type(&self) -> &AttributeType {
        &self.attribute_type
    }

    fn merge(&mut self, other: &Self) {
        match (self.data_type.clone(), other.data_type.clone()) {
            (DataType::Null, _) => {
                self.data_type = self.data_type.merge(&other.data_type);
                self.attribute_type = other.attribute_type;
            }
            (_, DataType::Null) => {
                self.data_type = self.data_type.merge(&other.data_type);
            }
            _ => {
                self.data_type = self.data_type.merge(&other.data_type);
                self.attribute_type = self.attribute_type.merge(&other.attribute_type);
            }
        }
    }
}

impl From<DataType> for AttributeDataType {
    fn from(value: DataType) -> Self {
        let attribute_type = AttributeType::infer(&value);

        Self {
            data_type: value,
            attribute_type,
        }
    }
}

impl From<(DataType, AttributeType)> for AttributeDataType {
    fn from(value: (DataType, AttributeType)) -> Self {
        Self {
            data_type: value.0,
            attribute_type: value.1,
        }
    }
}

enum AttributeSchemaKind<'a> {
    Node(&'a NodeIndex),
    Edge(&'a EdgeIndex),
}

impl AttributeSchemaKind<'_> {
    fn error_message(&self, key: &MedRecordAttribute, data_type: &DataType) -> String {
        match self {
            Self::Node(index) => {
                format!("Attribute {key} of type {data_type} not found on node with index {index}")
            }
            Self::Edge(index) => {
                format!("Attribute {key} of type {data_type} not found on edge with index {index}")
            }
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
                "Attribute {key} of node with index {index} is of type {data_type}. Expected {expected_data_type}."
            ),
            Self::Edge(index) => format!(
                "Attribute {key} of edge with index {index} is of type {data_type}. Expected {expected_data_type}."
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

type AttributeSchemaMapping = HashMap<MedRecordAttribute, AttributeDataType>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct AttributeSchema(AttributeSchemaMapping);

impl Deref for AttributeSchema {
    type Target = AttributeSchemaMapping;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> From<T> for AttributeSchema
where
    T: Into<AttributeSchemaMapping>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl AttributeSchema {
    pub fn new(mapping: HashMap<MedRecordAttribute, AttributeDataType>) -> Self {
        Self(mapping)
    }

    fn validate(
        &self,
        attributes: &Attributes,
        kind: AttributeSchemaKind,
    ) -> Result<(), GraphError> {
        for (key, schema) in &self.0 {
            let value = match (attributes.get(key), &schema.data_type) {
                (Some(value), _) => value,
                (None, DataType::Option(_)) => continue,
                (None, _) => {
                    return Err(GraphError::SchemaError(
                        kind.error_message(key, &schema.data_type),
                    ));
                }
            };

            let data_type = DataType::from(value);

            if !schema.data_type.evaluate(&data_type) {
                return Err(GraphError::SchemaError(kind.error_message_expected(
                    key,
                    &data_type,
                    &schema.data_type,
                )));
            }
        }

        let attributes_not_in_schema: Vec<_> = attributes
            .keys()
            .filter(|attribute| !self.0.contains_key(*attribute))
            .map(|attribute| attribute.to_string())
            .collect();

        if !attributes_not_in_schema.is_empty() {
            return Err(GraphError::SchemaError(
                kind.error_message_too_many(attributes_not_in_schema),
            ));
        }

        Ok(())
    }

    fn update(&mut self, attributes: &Attributes, empty: bool) {
        for (attribute, data_type) in self.0.iter_mut() {
            if !attributes.contains_key(attribute) {
                data_type.data_type = data_type.data_type.merge(&DataType::Null);
            }
        }

        for (attribute, value) in attributes {
            let data_type = DataType::from(value);
            let attribute_type = AttributeType::infer(&data_type);

            let mut attribute_data_type = AttributeDataType::new(data_type, attribute_type)
                .expect("AttributeType was inferred from DataType.");

            match self.0.entry(attribute.clone()) {
                Entry::Occupied(entry) => {
                    entry.into_mut().merge(&attribute_data_type);
                }
                Entry::Vacant(entry) => {
                    if !empty {
                        attribute_data_type.data_type =
                            attribute_data_type.data_type.merge(&DataType::Null);
                    }

                    entry.insert(attribute_data_type);
                }
            }
        }
    }

    fn infer(attributes: Vec<&Attributes>) -> Self {
        let mut schema = Self::default();

        let mut empty = true;

        for attributes in attributes {
            schema.update(attributes, empty);

            empty = false;
        }

        schema
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct GroupSchema {
    nodes: AttributeSchema,
    edges: AttributeSchema,
}

impl GroupSchema {
    pub fn new(nodes: AttributeSchema, edges: AttributeSchema) -> Self {
        Self { nodes, edges }
    }

    pub fn nodes(&self) -> &AttributeSchemaMapping {
        &self.nodes
    }

    pub fn edges(&self) -> &AttributeSchemaMapping {
        &self.edges
    }

    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        self.nodes
            .validate(attributes, AttributeSchemaKind::Node(index))
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
    ) -> Result<(), GraphError> {
        self.edges
            .validate(attributes, AttributeSchemaKind::Edge(index))
    }

    pub(crate) fn infer(nodes: Vec<&Attributes>, edges: Vec<&Attributes>) -> Self {
        Self {
            nodes: AttributeSchema::infer(nodes),
            edges: AttributeSchema::infer(edges),
        }
    }

    pub(crate) fn update_node(&mut self, attributes: &Attributes, empty: bool) {
        self.nodes.update(attributes, empty);
    }

    pub(crate) fn update_edge(&mut self, attributes: &Attributes, empty: bool) {
        self.edges.update(attributes, empty);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum SchemaType {
    Inferred,
    Provided,
}

impl Default for SchemaType {
    fn default() -> Self {
        Self::Inferred
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Schema {
    groups: HashMap<Group, GroupSchema>,
    ungrouped: GroupSchema,
    schema_type: SchemaType,
}

impl Schema {
    pub fn new_inferred(groups: HashMap<Group, GroupSchema>, ungrouped: GroupSchema) -> Self {
        Self {
            groups,
            ungrouped,
            schema_type: SchemaType::Inferred,
        }
    }

    pub fn new_provided(groups: HashMap<Group, GroupSchema>, ungrouped: GroupSchema) -> Self {
        Self {
            groups,
            ungrouped,
            schema_type: SchemaType::Provided,
        }
    }

    pub fn infer(medrecord: &MedRecord) -> Self {
        let mut group_mapping: MrHashMap<_, _> = medrecord
            .groups()
            .map(|group| (group, (Vec::new(), Vec::new())))
            .collect();

        let mut ungrouped = (Vec::new(), Vec::new());

        for node_index in medrecord.node_indices() {
            let mut groups_of_node = medrecord
                .groups_of_node(node_index)
                .expect("Node must exist.")
                .peekable();

            if groups_of_node.peek().is_none() {
                ungrouped.0.push(node_index);
                continue;
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
                ungrouped.1.push(edge_index);
                continue;
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
                    let node_attributes: Vec<_> = nodes_in_group
                        .into_iter()
                        .map(|node| medrecord.node_attributes(node).expect("Node must exist."))
                        .collect();
                    let edge_attributes: Vec<_> = edges_in_group
                        .into_iter()
                        .map(|edge| medrecord.edge_attributes(edge).expect("Edge must exist."))
                        .collect();

                    let schema = GroupSchema::infer(node_attributes, edge_attributes);

                    (group.clone(), schema)
                });

        let ungrouped_schema = GroupSchema::infer(
            ungrouped
                .0
                .into_iter()
                .map(|node| medrecord.node_attributes(node).expect("Node must exist."))
                .collect::<Vec<_>>(),
            ungrouped
                .1
                .into_iter()
                .map(|edge| medrecord.edge_attributes(edge).expect("Edge must exist."))
                .collect::<Vec<_>>(),
        );

        Self {
            groups: group_schemas.collect(),
            ungrouped: ungrouped_schema,
            schema_type: SchemaType::Inferred,
        }
    }

    pub fn groups(&self) -> &HashMap<Group, GroupSchema> {
        &self.groups
    }

    pub fn group(&self, group: &Group) -> Result<&GroupSchema, GraphError> {
        self.groups
            .get(group)
            .ok_or(GraphError::SchemaError(format!(
                "Group {group} not found in schema."
            )))
    }

    pub fn ungrouped(&self) -> &GroupSchema {
        &self.ungrouped
    }

    pub fn schema_type(&self) -> &SchemaType {
        &self.schema_type
    }

    pub fn validate_node<'a>(
        &self,
        index: &'a NodeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        match group {
            Some(group) => {
                let schema = self
                    .groups
                    .get(group)
                    .ok_or(GraphError::SchemaError(format!(
                        "Group {group} not found in schema."
                    )))?;

                schema.validate_node(index, attributes)
            }
            None => self.ungrouped.validate_node(index, attributes),
        }
    }

    pub fn validate_edge<'a>(
        &self,
        index: &'a EdgeIndex,
        attributes: &'a Attributes,
        group: Option<&'a Group>,
    ) -> Result<(), GraphError> {
        match group {
            Some(group) => {
                let schema = self
                    .groups
                    .get(group)
                    .ok_or(GraphError::SchemaError(format!(
                        "Group {group} not found in schema."
                    )))?;

                schema.validate_edge(index, attributes)
            }
            None => self.ungrouped.validate_edge(index, attributes),
        }
    }

    pub(crate) fn update_node(
        &mut self,
        attributes: &Attributes,
        group: Option<&Group>,
        empty: bool,
    ) {
        match group {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_node(attributes, empty);
            }
            None => self.ungrouped.update_node(attributes, empty),
        }
    }

    pub(crate) fn update_edge(
        &mut self,
        attributes: &Attributes,
        group: Option<&Group>,
        empty: bool,
    ) {
        match group {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_edge(attributes, empty);
            }
            None => self.ungrouped.update_edge(attributes, empty),
        }
    }

    pub fn set_node_attribute(
        &mut self,
        attribute: &MedRecordAttribute,
        data_type: DataType,
        attribute_type: AttributeType,
        group: Option<&Group>,
    ) -> Result<(), GraphError> {
        let attribute_data_type = AttributeDataType::new(data_type, attribute_type)?;

        match group {
            Some(group) => {
                let group_schema = self.groups.entry(group.clone()).or_default();
                group_schema
                    .nodes
                    .0
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
            None => {
                self.ungrouped
                    .nodes
                    .0
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
        }

        Ok(())
    }

    pub fn set_edge_attribute(
        &mut self,
        attribute: &MedRecordAttribute,
        data_type: DataType,
        attribute_type: AttributeType,
        group: Option<&Group>,
    ) -> Result<(), GraphError> {
        let attribute_data_type = AttributeDataType::new(data_type, attribute_type)?;

        match group {
            Some(group) => {
                let group_schema = self.groups.entry(group.clone()).or_default();
                group_schema
                    .edges
                    .0
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
            None => {
                self.ungrouped
                    .edges
                    .0
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
        }

        Ok(())
    }

    pub fn update_node_attribute(
        &mut self,
        attribute: &MedRecordAttribute,
        data_type: DataType,
        attribute_type: AttributeType,
        group: Option<&Group>,
    ) -> Result<(), GraphError> {
        let attribute_data_type = AttributeDataType::new(data_type, attribute_type)?;

        match group {
            Some(group) => {
                let group_schema = self.groups.entry(group.clone()).or_default();
                group_schema
                    .nodes
                    .0
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
            None => {
                self.ungrouped
                    .nodes
                    .0
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
        }

        Ok(())
    }

    pub fn update_edge_attribute(
        &mut self,
        attribute: &MedRecordAttribute,
        data_type: DataType,
        attribute_type: AttributeType,
        group: Option<&Group>,
    ) -> Result<(), GraphError> {
        let attribute_data_type = AttributeDataType::new(data_type, attribute_type)?;

        match group {
            Some(group) => {
                let group_schema = self.groups.entry(group.clone()).or_default();
                group_schema
                    .edges
                    .0
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
            None => {
                self.ungrouped
                    .edges
                    .0
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
        }

        Ok(())
    }

    pub fn remove_node_attribute(&mut self, attribute: &MedRecordAttribute, group: Option<&Group>) {
        match group {
            Some(group) => {
                if let Some(group_schema) = self.groups.get_mut(group) {
                    group_schema.nodes.0.remove(attribute);
                }
            }
            None => {
                self.ungrouped.nodes.0.remove(attribute);
            }
        }
    }

    pub fn remove_edge_attribute(&mut self, attribute: &MedRecordAttribute, group: Option<&Group>) {
        match group {
            Some(group) => {
                if let Some(group_schema) = self.groups.get_mut(group) {
                    group_schema.edges.0.remove(attribute);
                }
            }
            None => {
                self.ungrouped.edges.0.remove(attribute);
            }
        }
    }

    pub fn add_group(&mut self, group: Group, schema: GroupSchema) -> Result<(), GraphError> {
        if self.groups.contains_key(&group) {
            return Err(GraphError::SchemaError(format!(
                "Group {group} already exists in schema."
            )));
        }

        self.groups.insert(group, schema);

        Ok(())
    }

    pub fn remove_group(&mut self, group: &Group) {
        self.groups.remove(group);
    }

    pub fn freeze(&mut self) {
        self.schema_type = SchemaType::Provided;
    }

    pub fn unfreeze(&mut self) {
        self.schema_type = SchemaType::Inferred;
    }
}

#[cfg(test)]
mod test {
    use super::{AttributeDataType, GroupSchema};
    use crate::{
        medrecord::{
            datatypes::DataType,
            schema::{AttributeSchema, AttributeSchemaKind, AttributeType},
            Attributes, Schema, SchemaType,
        },
        MedRecord,
    };
    use std::collections::HashMap;

    #[test]
    fn test_attribute_type_infer() {
        assert_eq!(
            AttributeType::infer(&DataType::String),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::infer(&DataType::Int),
            AttributeType::Continuous
        );
        assert_eq!(
            AttributeType::infer(&DataType::Float),
            AttributeType::Continuous
        );
        assert_eq!(
            AttributeType::infer(&DataType::Bool),
            AttributeType::Categorical
        );
        assert_eq!(
            AttributeType::infer(&DataType::DateTime),
            AttributeType::Temporal
        );
        assert_eq!(
            AttributeType::infer(&DataType::Duration),
            AttributeType::Temporal
        );
        assert_eq!(
            AttributeType::infer(&DataType::Null),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::infer(&DataType::Any),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::infer(&DataType::Union((
                Box::new(DataType::Int),
                Box::new(DataType::Float)
            ))),
            AttributeType::Continuous
        );
        assert_eq!(
            AttributeType::infer(&DataType::Option(Box::new(DataType::Int))),
            AttributeType::Continuous
        );
    }

    #[test]
    fn test_attribute_type_merge() {
        assert_eq!(
            AttributeType::Categorical.merge(&AttributeType::Unstructured),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::Unstructured.merge(&AttributeType::Categorical),
            AttributeType::Unstructured
        );

        assert_eq!(
            AttributeType::Categorical.merge(&AttributeType::Categorical),
            AttributeType::Categorical
        );
        assert_eq!(
            AttributeType::Categorical.merge(&AttributeType::Continuous),
            AttributeType::Categorical
        );
        assert_eq!(
            AttributeType::Categorical.merge(&AttributeType::Temporal),
            AttributeType::Categorical
        );

        assert_eq!(
            AttributeType::Continuous.merge(&AttributeType::Categorical),
            AttributeType::Categorical
        );
        assert_eq!(
            AttributeType::Temporal.merge(&AttributeType::Categorical),
            AttributeType::Categorical
        );

        assert_eq!(
            AttributeType::Continuous.merge(&AttributeType::Continuous),
            AttributeType::Continuous
        );

        assert_eq!(
            AttributeType::Temporal.merge(&AttributeType::Temporal),
            AttributeType::Temporal
        );

        assert_eq!(
            AttributeType::Continuous.merge(&AttributeType::Temporal),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::Continuous.merge(&AttributeType::Unstructured),
            AttributeType::Unstructured
        );

        assert_eq!(
            AttributeType::Temporal.merge(&AttributeType::Continuous),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::Temporal.merge(&AttributeType::Unstructured),
            AttributeType::Unstructured
        );

        assert_eq!(
            AttributeType::Unstructured.merge(&AttributeType::Continuous),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::Unstructured.merge(&AttributeType::Temporal),
            AttributeType::Unstructured
        );
        assert_eq!(
            AttributeType::Unstructured.merge(&AttributeType::Unstructured),
            AttributeType::Unstructured
        );
    }

    #[test]
    fn test_data_type_merge() {
        assert_eq!(DataType::Int.merge(&DataType::Int), DataType::Int);
        assert_eq!(
            DataType::Int.merge(&DataType::Float),
            DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );
        assert_eq!(
            DataType::Int.merge(&DataType::Null),
            DataType::Option(Box::new(DataType::Int))
        );
        assert_eq!(
            DataType::Null.merge(&DataType::Int),
            DataType::Option(Box::new(DataType::Int))
        );
        assert_eq!(DataType::Null.merge(&DataType::Null), DataType::Null);
        assert_eq!(DataType::Int.merge(&DataType::Any), DataType::Any);
        assert_eq!(DataType::Any.merge(&DataType::Int), DataType::Any);
        assert_eq!(
            DataType::Option(Box::new(DataType::Int)).merge(&DataType::String),
            DataType::Option(Box::new(DataType::Union((
                Box::new(DataType::Int),
                Box::new(DataType::String)
            ))))
        );
        assert_eq!(
            DataType::Int.merge(&DataType::Option(Box::new(DataType::Int))),
            DataType::Option(Box::new(DataType::Int))
        );
        assert_eq!(
            DataType::Option(Box::new(DataType::Int))
                .merge(&DataType::Option(Box::new(DataType::String))),
            DataType::Option(Box::new(DataType::Union((
                Box::new(DataType::Int),
                Box::new(DataType::String)
            ))))
        );
    }

    #[test]
    fn test_attribute_data_type_new() {
        assert!(AttributeDataType::new(DataType::String, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::String, AttributeType::Continuous).is_err());
        assert!(AttributeDataType::new(DataType::String, AttributeType::Temporal).is_err());
        assert!(AttributeDataType::new(DataType::String, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Int, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Int, AttributeType::Continuous).is_ok());
        assert!(AttributeDataType::new(DataType::Int, AttributeType::Temporal).is_err());
        assert!(AttributeDataType::new(DataType::Int, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Float, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Float, AttributeType::Continuous).is_ok());
        assert!(AttributeDataType::new(DataType::Float, AttributeType::Temporal).is_err());
        assert!(AttributeDataType::new(DataType::Float, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Bool, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Bool, AttributeType::Continuous).is_err());
        assert!(AttributeDataType::new(DataType::Bool, AttributeType::Temporal).is_err());
        assert!(AttributeDataType::new(DataType::Bool, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::DateTime, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::DateTime, AttributeType::Continuous).is_err());
        assert!(AttributeDataType::new(DataType::DateTime, AttributeType::Temporal).is_ok());
        assert!(AttributeDataType::new(DataType::DateTime, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Duration, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Duration, AttributeType::Continuous).is_err());
        assert!(AttributeDataType::new(DataType::Duration, AttributeType::Temporal).is_ok());
        assert!(AttributeDataType::new(DataType::Duration, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Null, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Null, AttributeType::Continuous).is_ok());
        assert!(AttributeDataType::new(DataType::Null, AttributeType::Temporal).is_ok());
        assert!(AttributeDataType::new(DataType::Null, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(DataType::Any, AttributeType::Categorical).is_ok());
        assert!(AttributeDataType::new(DataType::Any, AttributeType::Continuous).is_err());
        assert!(AttributeDataType::new(DataType::Any, AttributeType::Temporal).is_err());
        assert!(AttributeDataType::new(DataType::Any, AttributeType::Unstructured).is_ok());

        assert!(AttributeDataType::new(
            DataType::Option(Box::new(DataType::Int)),
            AttributeType::Categorical
        )
        .is_ok());
        assert!(AttributeDataType::new(
            DataType::Option(Box::new(DataType::Int)),
            AttributeType::Continuous
        )
        .is_ok());
        assert!(AttributeDataType::new(
            DataType::Option(Box::new(DataType::Int)),
            AttributeType::Temporal
        )
        .is_err());
        assert!(AttributeDataType::new(
            DataType::Option(Box::new(DataType::Int)),
            AttributeType::Unstructured
        )
        .is_ok());

        assert!(AttributeDataType::new(
            DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float))),
            AttributeType::Categorical
        )
        .is_ok());
        assert!(AttributeDataType::new(
            DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float))),
            AttributeType::Continuous
        )
        .is_ok());
        assert!(AttributeDataType::new(
            DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float))),
            AttributeType::Temporal
        )
        .is_err());
        assert!(AttributeDataType::new(
            DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float))),
            AttributeType::Unstructured
        )
        .is_ok());
    }

    #[test]
    fn test_attribute_data_type_data_type() {
        let attribute_data_type = AttributeDataType::new(DataType::Int, AttributeType::Categorical)
            .expect("AttributeType was inferred from DataType.");

        assert_eq!(attribute_data_type.data_type(), &DataType::Int);
    }

    #[test]
    fn test_attribute_data_type_attribute_type() {
        let attribute_data_type = AttributeDataType::new(DataType::Int, AttributeType::Categorical)
            .expect("AttributeType was inferred from DataType.");

        assert_eq!(
            attribute_data_type.attribute_type(),
            &AttributeType::Categorical
        );
    }

    #[test]
    fn test_attribute_data_type_merge() {
        let mut attribute_data_type =
            AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                .expect("AttributeType was inferred from DataType.");

        attribute_data_type.merge(
            &AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                .expect("AttributeType was inferred from DataType."),
        );

        assert_eq!(
            attribute_data_type.data_type(),
            &DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );
        assert_eq!(
            attribute_data_type.attribute_type(),
            &AttributeType::Categorical
        );
    }

    #[test]
    fn test_attribute_data_type_from_data_type() {
        let attribute_data_type: AttributeDataType = DataType::Int.into();

        assert_eq!(attribute_data_type.data_type(), &DataType::Int);
        assert_eq!(
            attribute_data_type.attribute_type(),
            &AttributeType::Continuous
        );
    }

    #[test]
    fn test_attribute_data_type_from_tuple() {
        let attribute_data_type: AttributeDataType =
            (DataType::Int, AttributeType::Categorical).into();

        assert_eq!(attribute_data_type.data_type(), &DataType::Int);
        assert_eq!(
            attribute_data_type.attribute_type(),
            &AttributeType::Categorical
        );
    }

    #[test]
    fn test_attribute_schema_kind_error_message() {
        let index = 0;
        let key = "key";
        let data_type = DataType::Int;

        assert_eq!(
            AttributeSchemaKind::Node(&(index.into())).error_message(&(key.into()), &data_type),
            "Attribute key of type Int not found on node with index 0"
        );
        assert_eq!(
            AttributeSchemaKind::Edge(&(index as u32)).error_message(&(key.into()), &data_type),
            "Attribute key of type Int not found on edge with index 0"
        );
    }

    #[test]
    fn test_attribute_schema_kind_error_message_expected() {
        let index = 0;
        let key = "key";
        let data_type = DataType::Int;
        let expected_data_type = DataType::Float;

        assert_eq!(
            AttributeSchemaKind::Node(&(index.into())).error_message_expected(
                &(key.into()),
                &data_type,
                &expected_data_type
            ),
            "Attribute key of node with index 0 is of type Int. Expected Float."
        );
        assert_eq!(
            AttributeSchemaKind::Edge(&(index as u32)).error_message_expected(
                &(key.into()),
                &data_type,
                &expected_data_type
            ),
            "Attribute key of edge with index 0 is of type Int. Expected Float."
        );
    }

    #[test]
    fn test_attribute_schema_kind_error_message_too_many() {
        let index = 0;
        let attributes = vec!["key1".to_string(), "key2".to_string()];

        assert_eq!(
            AttributeSchemaKind::Node(&(index.into())).error_message_too_many(attributes.clone()),
            "Attributes [key1, key2] of node with index 0 do not exist in schema."
        );
        assert_eq!(
            AttributeSchemaKind::Edge(&(index as u32)).error_message_too_many(attributes),
            "Attributes [key1, key2] of edge with index 0 do not exist in schema."
        );
    }

    #[test]
    fn test_attribute_schema_deref() {
        let schema = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        assert_eq!(
            schema.get(&"key1".into()).unwrap().data_type(),
            &DataType::Int
        );
        assert_eq!(
            schema.get(&"key2".into()).unwrap().data_type(),
            &DataType::Float
        );
    }

    #[test]
    fn test_attribute_schema_validate() {
        let attribute_schema = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let attributes: Attributes = vec![("key1".into(), 0.into()), ("key2".into(), 0.0.into())]
            .into_iter()
            .collect();

        assert!(attribute_schema
            .validate(&attributes, AttributeSchemaKind::Node(&0.into()))
            .is_ok());

        let attributes: Attributes = vec![("key1".into(), 0.0.into()), ("key2".into(), 0.into())]
            .into_iter()
            .collect();

        assert!(attribute_schema
            .validate(&attributes, AttributeSchemaKind::Node(&0.into()))
            .is_err_and(|error| { matches!(error, crate::errors::GraphError::SchemaError(_)) }));

        let attributes: Attributes = vec![
            ("key1".into(), 0.into()),
            ("key2".into(), 0.0.into()),
            ("key3".into(), 0.0.into()),
        ]
        .into_iter()
        .collect();

        assert!(attribute_schema
            .validate(&attributes, AttributeSchemaKind::Node(&0.into()))
            .is_err_and(|error| { matches!(error, crate::errors::GraphError::SchemaError(_)) }));
    }

    #[test]
    fn test_attribute_schema_update() {
        let mut schema = AttributeSchema::default();
        let attributes: Attributes =
            vec![("key1".into(), 0.into()), ("key2".into(), "test".into())]
                .into_iter()
                .collect();

        schema.update(&attributes, true);

        assert_eq!(schema.0.len(), 2);
        assert_eq!(
            schema.0.get(&"key1".into()).unwrap().data_type(),
            &DataType::Int
        );
        assert_eq!(
            schema.0.get(&"key2".into()).unwrap().data_type(),
            &DataType::String
        );

        let new_attributes: Attributes =
            vec![("key1".into(), 0.5.into()), ("key3".into(), true.into())]
                .into_iter()
                .collect();

        schema.update(&new_attributes, false);

        assert_eq!(schema.0.len(), 3);
        assert_eq!(
            schema.0.get(&"key1".into()).unwrap().data_type(),
            &DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );
        assert_eq!(
            schema.0.get(&"key2".into()).unwrap().data_type(),
            &DataType::Option(Box::new(DataType::String))
        );
        assert_eq!(
            schema.0.get(&"key3".into()).unwrap().data_type(),
            &DataType::Option(Box::new(DataType::Bool))
        );
    }

    #[test]
    fn test_attribute_schema_infer() {
        let attributes1: Attributes =
            vec![("key1".into(), 0.into()), ("key2".into(), "test".into())]
                .into_iter()
                .collect();

        let attributes2: Attributes = vec![("key1".into(), 1.into()), ("key3".into(), true.into())]
            .into_iter()
            .collect();

        let schema = AttributeSchema::infer(vec![&attributes1, &attributes2]);

        assert_eq!(schema.0.len(), 3);
        assert_eq!(
            schema.0.get(&"key1".into()).unwrap().data_type(),
            &DataType::Int
        );
        assert_eq!(
            schema.0.get(&"key2".into()).unwrap().data_type(),
            &DataType::Option(Box::new(DataType::String))
        );
        assert_eq!(
            schema.0.get(&"key3".into()).unwrap().data_type(),
            &DataType::Option(Box::new(DataType::Bool))
        );
    }

    #[test]
    fn test_group_schema_nodes() {
        let nodes = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let group_schema = GroupSchema::new(nodes.clone(), AttributeSchema::default());

        assert_eq!(group_schema.nodes(), &nodes.0);
    }

    #[test]
    fn test_group_schema_edges() {
        let edges = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let group_schema = GroupSchema::new(AttributeSchema::default(), edges.clone());

        assert_eq!(group_schema.edges(), &edges.0);
    }

    #[test]
    fn test_group_schema_validate_node() {
        let nodes = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let group_schema = GroupSchema::new(nodes, AttributeSchema::default());

        let attributes: Attributes = vec![("key1".into(), 0.into()), ("key2".into(), 0.0.into())]
            .into_iter()
            .collect();

        assert!(group_schema.validate_node(&0.into(), &attributes).is_ok());

        let attributes: Attributes = vec![("key1".into(), 0.0.into()), ("key2".into(), 0.into())]
            .into_iter()
            .collect();

        assert!(group_schema
            .validate_node(&0.into(), &attributes)
            .is_err_and(|error| { matches!(error, crate::errors::GraphError::SchemaError(_)) }));
    }

    #[test]
    fn test_group_schema_validate_edge() {
        let edges = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let group_schema = GroupSchema::new(AttributeSchema::default(), edges);

        let attributes: Attributes = vec![("key1".into(), 0.into()), ("key2".into(), 0.0.into())]
            .into_iter()
            .collect();

        assert!(group_schema.validate_edge(&0, &attributes).is_ok());

        let attributes: Attributes = vec![("key1".into(), 0.0.into()), ("key2".into(), 0.into())]
            .into_iter()
            .collect();

        assert!(group_schema
            .validate_edge(&0, &attributes)
            .is_err_and(|error| { matches!(error, crate::errors::GraphError::SchemaError(_)) }));
    }

    #[test]
    fn test_group_schema_infer() {
        let node_attributes1: Attributes =
            vec![("key1".into(), 0.into()), ("key2".into(), "test".into())]
                .into_iter()
                .collect();

        let node_attributes2: Attributes =
            vec![("key1".into(), 1.into()), ("key3".into(), true.into())]
                .into_iter()
                .collect();

        let edge_attributes: Attributes =
            vec![("key4".into(), 0.5.into()), ("key5".into(), "edge".into())]
                .into_iter()
                .collect();

        let group_schema = GroupSchema::infer(
            vec![&node_attributes1, &node_attributes2],
            vec![&edge_attributes],
        );

        assert_eq!(group_schema.nodes().len(), 3);
        assert_eq!(group_schema.edges().len(), 2);

        assert_eq!(
            group_schema
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Int
        );
        assert_eq!(
            group_schema
                .nodes()
                .get(&"key2".into())
                .unwrap()
                .data_type(),
            &DataType::Option(Box::new(DataType::String))
        );
        assert_eq!(
            group_schema
                .nodes()
                .get(&"key3".into())
                .unwrap()
                .data_type(),
            &DataType::Option(Box::new(DataType::Bool))
        );

        assert_eq!(
            group_schema
                .edges()
                .get(&"key4".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );
        assert_eq!(
            group_schema
                .edges()
                .get(&"key5".into())
                .unwrap()
                .data_type(),
            &DataType::String
        );
    }

    #[test]
    fn test_group_schema_update_node() {
        let mut group_schema = GroupSchema::default();
        let attributes = Attributes::from([("key1".into(), 0.into()), ("key2".into(), 0.0.into())]);

        group_schema.update_node(&attributes, true);

        assert_eq!(group_schema.nodes().len(), 2);
        assert_eq!(
            group_schema
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Int
        );
        assert_eq!(
            group_schema
                .nodes()
                .get(&"key2".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );
    }

    #[test]
    fn test_group_schema_update_edge() {
        let mut group_schema = GroupSchema::default();
        let attributes =
            Attributes::from([("key3".into(), true.into()), ("key4".into(), "test".into())]);

        group_schema.update_edge(&attributes, true);

        assert_eq!(group_schema.edges().len(), 2);
        assert_eq!(
            group_schema
                .edges()
                .get(&"key3".into())
                .unwrap()
                .data_type(),
            &DataType::Bool
        );
        assert_eq!(
            group_schema
                .edges()
                .get(&"key4".into())
                .unwrap()
                .data_type(),
            &DataType::String
        );
    }

    #[test]
    fn test_schema_infer() {
        let mut medrecord = MedRecord::new();
        medrecord
            .add_node(0.into(), Attributes::from([("key1".into(), 0.into())]))
            .unwrap();
        medrecord
            .add_node(1.into(), Attributes::from([("key2".into(), 0.0.into())]))
            .unwrap();
        medrecord
            .add_edge(
                0.into(),
                1.into(),
                Attributes::from([("key3".into(), true.into())]),
            )
            .unwrap();

        let schema = Schema::infer(&medrecord);

        assert_eq!(schema.ungrouped().nodes().len(), 2);
        assert_eq!(schema.ungrouped().edges().len(), 1);

        medrecord
            .add_group("test".into(), Some(vec![0.into(), 1.into()]), Some(vec![0]))
            .unwrap();

        let schema = Schema::infer(&medrecord);

        assert_eq!(schema.groups().len(), 1);
        assert_eq!(schema.group(&"test".into()).unwrap().nodes().len(), 2);
        assert_eq!(schema.group(&"test".into()).unwrap().edges().len(), 1);
    }

    #[test]
    fn test_schema_groups() {
        let schema = Schema::new_inferred(
            vec![("group1".into(), GroupSchema::default())]
                .into_iter()
                .collect(),
            GroupSchema::default(),
        );
        assert_eq!(schema.groups().len(), 1);
        assert!(schema.groups().contains_key(&"group1".into()));
    }

    #[test]
    fn test_schema_group() {
        let schema = Schema::new_inferred(
            vec![("group1".into(), GroupSchema::default())]
                .into_iter()
                .collect(),
            GroupSchema::default(),
        );
        assert!(schema.group(&"group1".into()).is_ok());
        assert!(schema.group(&"non_existent".into()).is_err());
    }

    #[test]
    fn test_schema_default() {
        let default_schema = GroupSchema::default();
        let schema = Schema::new_inferred(HashMap::new(), default_schema.clone());
        assert_eq!(schema.ungrouped(), &default_schema);
    }

    #[test]
    fn test_schema_schema_type() {
        let schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        assert_eq!(schema.schema_type(), &SchemaType::Inferred);
    }

    #[test]
    fn test_schema_validate_node() {
        let mut schema = Schema::new_inferred(
            HashMap::new(),
            GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
        );
        schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                None,
            )
            .unwrap();

        let attributes = Attributes::from([("key1".into(), 0.into())]);
        assert!(schema.validate_node(&0.into(), &attributes, None).is_ok());

        let invalid_attributes = Attributes::from([("key1".into(), "invalid".into())]);
        assert!(schema
            .validate_node(&0.into(), &invalid_attributes, None)
            .is_err());
    }

    #[test]
    fn test_schema_validate_edge() {
        let mut schema = Schema::new_inferred(
            HashMap::new(),
            GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
        );
        schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                None,
            )
            .unwrap();

        let attributes = Attributes::from([("key1".into(), true.into())]);
        assert!(schema.validate_edge(&0, &attributes, None).is_ok());

        let invalid_attributes = Attributes::from([("key1".into(), 0.into())]);
        assert!(schema.validate_edge(&0, &invalid_attributes, None).is_err());
    }

    #[test]
    fn test_schema_update_node() {
        let mut schema = Schema::new_inferred(
            HashMap::new(),
            GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
        );
        let attributes = Attributes::from([("key1".into(), 0.into()), ("key2".into(), 0.0.into())]);

        schema.update_node(&attributes, None, true);

        assert_eq!(schema.ungrouped().nodes().len(), 2);
        assert_eq!(
            schema
                .ungrouped()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Int
        );
        assert_eq!(
            schema
                .ungrouped()
                .nodes()
                .get(&"key2".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );
    }

    #[test]
    fn test_schema_update_edge() {
        let mut schema = Schema::new_inferred(
            HashMap::new(),
            GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
        );
        let attributes =
            Attributes::from([("key3".into(), true.into()), ("key4".into(), "test".into())]);

        schema.update_edge(&attributes, None, true);

        assert_eq!(schema.ungrouped().edges().len(), 2);
        assert_eq!(
            schema
                .ungrouped()
                .edges()
                .get(&"key3".into())
                .unwrap()
                .data_type(),
            &DataType::Bool
        );
        assert_eq!(
            schema
                .ungrouped()
                .edges()
                .get(&"key4".into())
                .unwrap()
                .data_type(),
            &DataType::String
        );
    }

    #[test]
    fn test_schema_set_node_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        assert!(schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Int
        );
        assert!(schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );

        assert!(schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                Some(&"group1".into())
            )
            .is_ok());
        assert_eq!(
            schema
                .group(&"group1".into())
                .unwrap()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );
    }

    #[test]
    fn test_schema_set_edge_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        assert!(schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .edges()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Bool
        );
        assert!(schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .edges()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );

        assert!(schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                Some(&"group1".into())
            )
            .is_ok());
        assert_eq!(
            schema
                .group(&"group1".into())
                .unwrap()
                .edges()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Float
        );
    }

    #[test]
    fn test_schema_update_node_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                None,
            )
            .unwrap();
        assert!(schema
            .update_node_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );

        schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                Some(&"group1".into()),
            )
            .unwrap();
        assert!(schema
            .update_node_attribute(
                &"key1".into(),
                DataType::Float,
                AttributeType::Continuous,
                Some(&"group1".into())
            )
            .is_ok());
        assert_eq!(
            schema
                .group(&"group1".into())
                .unwrap()
                .nodes()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );
    }

    #[test]
    fn test_schema_update_edge_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                None,
            )
            .unwrap();
        assert!(schema
            .update_edge_attribute(
                &"key1".into(),
                DataType::String,
                AttributeType::Unstructured,
                None
            )
            .is_ok());
        assert_eq!(
            schema
                .ungrouped()
                .edges()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Union((Box::new(DataType::Bool), Box::new(DataType::String)))
        );

        schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                Some(&"group1".into()),
            )
            .unwrap();
        assert!(schema
            .update_edge_attribute(
                &"key1".into(),
                DataType::String,
                AttributeType::Unstructured,
                Some(&"group1".into())
            )
            .is_ok());
        assert_eq!(
            schema
                .group(&"group1".into())
                .unwrap()
                .edges()
                .get(&"key1".into())
                .unwrap()
                .data_type(),
            &DataType::Union((Box::new(DataType::Bool), Box::new(DataType::String)))
        );
    }

    #[test]
    fn test_schema_remove_node_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                None,
            )
            .unwrap();
        schema.remove_node_attribute(&"key1".into(), None);
        assert!(!schema.ungrouped().nodes().contains_key(&"key1".into()));

        schema
            .set_node_attribute(
                &"key1".into(),
                DataType::Int,
                AttributeType::Continuous,
                Some(&"group1".into()),
            )
            .unwrap();
        schema.remove_node_attribute(&"key1".into(), Some(&"group1".into()));
        assert!(!schema
            .group(&"group1".into())
            .unwrap()
            .nodes()
            .contains_key(&"key1".into()));
    }

    #[test]
    fn test_schema_remove_edge_attribute() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                None,
            )
            .unwrap();
        schema.remove_edge_attribute(&"key1".into(), None);
        assert!(!schema.ungrouped().edges().contains_key(&"key1".into()));

        schema
            .set_edge_attribute(
                &"key1".into(),
                DataType::Bool,
                AttributeType::Categorical,
                Some(&"group1".into()),
            )
            .unwrap();
        schema.remove_edge_attribute(&"key1".into(), Some(&"group1".into()));
        assert!(!schema
            .group(&"group1".into())
            .unwrap()
            .edges()
            .contains_key(&"key1".into()));
    }

    #[test]
    fn test_schema_add_group() {
        let attribute_schema = AttributeSchema::new(
            vec![
                (
                    "key1".into(),
                    AttributeDataType::new(DataType::Int, AttributeType::Categorical)
                        .expect("AttributeType was inferred from DataType."),
                ),
                (
                    "key2".into(),
                    AttributeDataType::new(DataType::Float, AttributeType::Continuous)
                        .expect("AttributeType was inferred from DataType."),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        schema
            .add_group(
                "group1".into(),
                GroupSchema::new(attribute_schema.clone(), AttributeSchema::default()),
            )
            .unwrap();
        assert_eq!(
            attribute_schema,
            schema.group(&"group1".into()).unwrap().nodes
        );

        assert!(schema
            .add_group("group1".into(), GroupSchema::default())
            .is_err_and(|error| { matches!(error, crate::errors::GraphError::SchemaError(_)) }));
    }

    #[test]
    fn test_schema_remove_group() {
        let mut schema = Schema::new_inferred(
            vec![("group1".into(), GroupSchema::default())]
                .into_iter()
                .collect(),
            GroupSchema::default(),
        );
        schema.remove_group(&"group1".into());
        assert!(!schema.groups().contains_key(&"group1".into()));
    }

    #[test]
    fn test_schema_freeze_unfreeze() {
        let mut schema = Schema::new_inferred(HashMap::new(), GroupSchema::default());
        assert_eq!(schema.schema_type(), &SchemaType::Inferred);

        schema.freeze();
        assert_eq!(schema.schema_type(), &SchemaType::Provided);

        schema.unfreeze();
        assert_eq!(schema.schema_type(), &SchemaType::Inferred);
    }
}

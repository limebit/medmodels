use super::{Attributes, EdgeIndex, Group, MedRecord, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, MedRecordAttribute},
};
use medmodels_utils::aliases::MrHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::Entry, HashMap};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
            DataType::Duration => Self::Continuous,
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
            (Self::Categorical, Self::Categorical) => Self::Categorical,
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

            (AttributeType::Continuous, DataType::Int | DataType::Float) => Ok(()),
            (AttributeType::Continuous, _) => Err(GraphError::SchemaError(
                "Continuous attribute must be of (sub-)type Int or Float.".to_string(),
            )),

            (AttributeType::Temporal, DataType::DateTime | DataType::Duration) => Ok(()),
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
        self.data_type = self.data_type.merge(&other.data_type);
        self.attribute_type = self.attribute_type.merge(&other.attribute_type);
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

type AttributeSchema = HashMap<MedRecordAttribute, AttributeDataType>;

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct GroupSchema {
    nodes: AttributeSchema,
    edges: AttributeSchema,
}

impl GroupSchema {
    pub fn new(nodes: AttributeSchema, edges: AttributeSchema) -> Self {
        Self { nodes, edges }
    }

    pub fn nodes(&self) -> &AttributeSchema {
        &self.nodes
    }

    pub fn edges(&self) -> &AttributeSchema {
        &self.edges
    }

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

    pub(crate) fn infer(nodes: Vec<&Attributes>, edges: Vec<&Attributes>) -> Self {
        Self {
            nodes: Self::infer_attribute_schema(nodes),
            edges: Self::infer_attribute_schema(edges),
        }
    }

    pub(crate) fn update_node(&mut self, attributes: &Attributes) {
        Self::update_attribute_schema(attributes, &mut self.nodes);
    }

    pub(crate) fn update_edge(&mut self, attributes: &Attributes) {
        Self::update_attribute_schema(attributes, &mut self.edges);
    }

    fn infer_attribute_schema(attributes: Vec<&Attributes>) -> AttributeSchema {
        let mut schema = AttributeSchema::new();

        for attributes in attributes {
            Self::update_attribute_schema(attributes, &mut schema);
        }

        schema
    }

    fn update_attribute_schema(attributes: &Attributes, schema: &mut AttributeSchema) {
        for (attribute, value) in attributes {
            let data_type = DataType::from(value);
            let attribute_type = AttributeType::infer(&data_type);

            let attribute_data_type = AttributeDataType::new(data_type, attribute_type)
                .expect("AttributeType was infered from DataType.");

            match schema.entry(attribute.clone()) {
                Entry::Occupied(entry) => {
                    entry.into_mut().merge(&attribute_data_type);
                }
                Entry::Vacant(entry) => {
                    entry.insert(attribute_data_type);
                }
            }
        }

        for (attribute, data_type) in schema {
            if !attributes.contains_key(attribute) {
                data_type.data_type = data_type.data_type.merge(&DataType::Null);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
    default: GroupSchema,
    schema_type: SchemaType,
}

impl Schema {
    pub fn new_inferred(groups: HashMap<Group, GroupSchema>, default: GroupSchema) -> Self {
        Self {
            groups,
            default,
            schema_type: SchemaType::Inferred,
        }
    }

    pub fn new_provided(groups: HashMap<Group, GroupSchema>, default: GroupSchema) -> Self {
        Self {
            groups,
            default,
            schema_type: SchemaType::Provided,
        }
    }

    pub fn infer(medrecord: &MedRecord) -> Self {
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
            default: default_schema,
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
                "Group {} not found in schema.",
                group
            )))
    }

    pub fn default(&self) -> &GroupSchema {
        &self.default
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

    pub(crate) fn update_node(&mut self, attributes: &Attributes, group: Option<&Group>) {
        match group {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_node(attributes);
            }
            None => self.default.update_node(attributes),
        }
    }

    pub(crate) fn update_edge(&mut self, attributes: &Attributes, group: Option<&Group>) {
        match group {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_edge(attributes);
            }
            None => self.default.update_edge(attributes),
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
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
            None => {
                self.default
                    .nodes
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
                    .insert(attribute.clone(), attribute_data_type.clone());
            }
            None => {
                self.default
                    .edges
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
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
            None => {
                self.default
                    .nodes
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
                    .entry(attribute.clone())
                    .and_modify(|value| value.merge(&attribute_data_type))
                    .or_insert(attribute_data_type);
            }
            None => {
                self.default
                    .edges
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
                    group_schema.nodes.remove(attribute);
                }
            }
            None => {
                self.default.nodes.remove(attribute);
            }
        }
    }

    pub fn remove_edge_attribute(&mut self, attribute: &MedRecordAttribute, group: Option<&Group>) {
        match group {
            Some(group) => {
                if let Some(group_schema) = self.groups.get_mut(group) {
                    group_schema.edges.remove(attribute);
                }
            }
            None => {
                self.default.edges.remove(attribute);
            }
        }
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

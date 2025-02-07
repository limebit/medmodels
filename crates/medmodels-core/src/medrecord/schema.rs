use super::{Attributes, EdgeIndex, Group, MedRecord, NodeIndex};
use crate::{
    errors::GraphError,
    medrecord::{datatypes::DataType, MedRecordAttribute},
};
use medmodels_utils::aliases::MrHashMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttributeType {
    Categorical,
    Continuous,
    Temporal,
    Unstructured,
}

impl AttributeType {
    pub fn infer_from(data_type: &DataType) -> Self {
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
                Self::infer_from(first_dataype).merge(&Self::infer_from(second_dataype))
            }
            DataType::Option(dataype) => Self::infer_from(dataype),
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
    pub fn new(data_type: DataType, attribute_type: AttributeType) -> Self {
        Self {
            data_type,
            attribute_type,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.data_type = self.data_type.merge(&other.data_type);
        self.attribute_type = self.attribute_type.merge(&other.attribute_type);
    }
}

impl From<DataType> for AttributeDataType {
    fn from(value: DataType) -> Self {
        let attribute_type = AttributeType::infer_from(&value);

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
            let attribute_type = AttributeType::infer_from(&data_type);

            let attribute_data_type = AttributeDataType::new(data_type, attribute_type);

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Schema {
    groups: HashMap<Group, Arc<GroupSchema>>,
    default: Arc<GroupSchema>,
    pub(crate) schema_type: SchemaType,
}

impl Default for Schema {
    fn default() -> Self {
        Self {
            groups: HashMap::new(),
            default: Arc::new(GroupSchema::default()),
            schema_type: SchemaType::default(),
        }
    }
}

impl Schema {
    pub fn new_inferred(groups: HashMap<Group, GroupSchema>, default: GroupSchema) -> Self {
        Self {
            groups: groups
                .into_iter()
                .map(|(group, schema)| (group, Arc::new(schema)))
                .collect(),
            default: Arc::new(default),
            schema_type: SchemaType::Inferred,
        }
    }

    pub fn new_provided(groups: HashMap<Group, GroupSchema>, default: GroupSchema) -> Self {
        Self {
            groups: groups
                .into_iter()
                .map(|(group, schema)| (group, Arc::new(schema)))
                .collect(),
            default: Arc::new(default),
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

                    (group.clone(), Arc::new(schema))
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
            default: Arc::new(default_schema),
            schema_type: SchemaType::Inferred,
        }
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

    pub(crate) fn update_node(&mut self, attributes: &Attributes, groups: Option<&Group>) {
        match groups {
            Some(group) => {
                Arc::make_mut(&mut self.groups.entry(group.clone()).or_default())
                    .update_node(attributes);
            }
            None => Arc::make_mut(&mut self.default).update_node(attributes),
        }
    }

    pub(crate) fn update_edge(&mut self, attributes: &Attributes, groups: Option<&Group>) {
        match groups {
            Some(group) => {
                Arc::make_mut(&mut self.groups.entry(group.clone()).or_default())
                    .update_edge(attributes);
            }
            None => Arc::make_mut(&mut self.default).update_edge(attributes),
        }
    }
}

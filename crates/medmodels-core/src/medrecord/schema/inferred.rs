use super::{
    provided::{ProvidedGroupSchema, ProvidedSchema},
    AttributeDataType, AttributeSchema, AttributeType, Attributes, MedRecord,
};
use crate::medrecord::{datatypes::DataType, Group};
use medmodels_utils::aliases::MrHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::Entry, HashMap};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct InferredGroupSchema {
    pub nodes: AttributeSchema,
    pub edges: AttributeSchema,
}

impl From<ProvidedGroupSchema> for InferredGroupSchema {
    fn from(value: ProvidedGroupSchema) -> Self {
        Self {
            nodes: value.nodes,
            edges: value.edges,
        }
    }
}

impl InferredGroupSchema {
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
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct InferredSchema {
    pub groups: HashMap<Group, InferredGroupSchema>,
    pub default: InferredGroupSchema,
}

impl From<ProvidedSchema> for InferredSchema {
    fn from(value: ProvidedSchema) -> Self {
        Self {
            groups: value
                .groups
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            default: value.default.into(),
        }
    }
}

impl InferredSchema {
    pub fn infer(medrecord: &MedRecord) -> InferredSchema {
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

                    let schema = InferredGroupSchema::infer(node_attributes, edge_attributes);

                    (group.clone(), schema)
                });

        let default_schema = InferredGroupSchema::infer(
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
        }
    }

    pub(crate) fn update_node(&mut self, attributes: &Attributes, groups: Option<&Group>) {
        match groups {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_node(attributes);
            }
            None => self.default.update_node(attributes),
        }
    }

    pub(crate) fn update_edge(&mut self, attributes: &Attributes, groups: Option<&Group>) {
        match groups {
            Some(group) => {
                self.groups
                    .entry(group.clone())
                    .or_default()
                    .update_edge(attributes);
            }
            None => self.default.update_edge(attributes),
        }
    }
}

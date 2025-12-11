#![allow(clippy::type_complexity)]

mod tabled_modifiers;

use crate::{
    errors::MedRecordError,
    medrecord::{
        overview::tabled_modifiers::MergeDuplicatesVerticalByColumn,
        querying::{
            edges::EdgeOperand,
            nodes::NodeOperand,
            wrapper::{MatchMode, Wrapper},
        },
    },
    prelude::{AttributeType, DataType, Group, GroupSchema, MedRecordAttribute, MedRecordValue},
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashMap;
use std::{
    cmp::Ordering,
    fmt::{Display, Formatter},
};
use tabled::{
    builder::Builder,
    settings::{themes::BorderCorrection, Alignment, Panel, Style},
};

#[derive(Debug, Clone)]
pub enum AttributeOverviewData {
    Categorical {
        distinct_values: Vec<MedRecordValue>,
    },
    Continuous {
        min: MedRecordValue,
        mean: MedRecordValue,
        max: MedRecordValue,
    },
    Temporal {
        min: MedRecordValue,
        max: MedRecordValue,
    },
    Unstructured {
        distinct_count: usize,
    },
}

impl AttributeOverviewData {
    fn attribute_type_name(&self) -> &'static str {
        match self {
            Self::Categorical { .. } => "Categorical",
            Self::Continuous { .. } => "Continuous",
            Self::Temporal { .. } => "Temporal",
            Self::Unstructured { .. } => "Unstructured",
        }
    }

    fn details(&self) -> String {
        match self {
            Self::Categorical { distinct_values } => {
                format!(
                    "Distinct values: [{}]",
                    distinct_values.iter().map(|v| v.to_string()).join(", ")
                )
            }
            Self::Continuous { min, mean, max } => {
                format!("Min: {}\nMean: {}\nMax: {}", min, mean, max)
            }
            Self::Temporal { min, max } => {
                format!("Min: {}\nMax: {}", min, max)
            }
            Self::Unstructured { distinct_count } => {
                format!("Distinct value count: {}", distinct_count)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttributeOverview {
    pub data_type: DataType,
    pub data: AttributeOverviewData,
}

#[derive(Debug, Clone)]
pub struct NodeGroupOverview {
    pub count: usize,
    pub attributes: MrHashMap<MedRecordAttribute, AttributeOverview>,
}

impl Display for NodeGroupOverview {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut builder = Builder::new();

        builder.push_record([
            "Node Count",
            "Attribute",
            "Attribute Type",
            "Data Type",
            "Details",
        ]);

        for (attribute, overview) in &self.attributes {
            let details = overview.data.details();

            builder.push_record([
                &self.count.to_string(),
                &attribute.to_string(),
                overview.data.attribute_type_name(),
                &overview.data_type.to_string(),
                &details,
            ]);
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Node Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        writeln!(f, "{table}")
    }
}

impl NodeGroupOverview {
    fn new(
        medrecord: &MedRecord,
        group_schema: &GroupSchema,
        group: Option<&Group>,
    ) -> Result<Self, MedRecordError> {
        let node_selection_query: Box<dyn Fn(&Wrapper<NodeOperand>)> = match group {
            Some(group) => Box::new(|nodes| {
                nodes.in_group(group.clone());
            }),
            None => Box::new(|nodes| {
                nodes.exclude(|nodes| {
                    let groups: Vec<_> = medrecord.groups().cloned().collect();

                    nodes.in_group((groups, MatchMode::Any));
                });
            }),
        };

        let count = medrecord
            .query_nodes(|nodes| {
                node_selection_query(nodes);

                nodes.index().count()
            })
            .evaluate()?
            .unwrap_or(MedRecordAttribute::Int(0));

        let count = match count {
            MedRecordAttribute::Int(c) => c as usize,
            _ => unreachable!(),
        };

        let attributes: MrHashMap<_, _> = group_schema
            .nodes()
            .iter()
            .map(|(key, attribute_data_type)| {
                let attribute_type = attribute_data_type.attribute_type();
                let data_type = attribute_data_type.data_type().clone();

                let attribute_overview = match attribute_type {
                    AttributeType::Categorical => {
                        let values = medrecord
                            .query_nodes(|nodes| {
                                node_selection_query(nodes);

                                nodes.attribute(key.clone())
                            })
                            .evaluate()?
                            .map(|(_, value)| value)
                            .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .dedup_by(|a, b| a == b)
                            .collect();

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Categorical {
                                distinct_values: values,
                            },
                        }
                    }
                    AttributeType::Continuous => {
                        let (min, mean, max) = medrecord
                            .query_nodes(|nodes| {
                                node_selection_query(nodes);

                                let values = nodes.attribute(key.clone());

                                (values.min(), values.mean(), values.max())
                            })
                            .evaluate()?;

                        let min = min.map(|min| min.1).unwrap_or(MedRecordValue::Null);
                        let mean = mean.unwrap_or(MedRecordValue::Null);
                        let max = max.map(|max| max.1).unwrap_or(MedRecordValue::Null);

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Continuous { min, mean, max },
                        }
                    }
                    AttributeType::Temporal => {
                        let (min, max) = medrecord
                            .query_nodes(|nodes| {
                                node_selection_query(nodes);

                                let values = nodes.attribute(key.clone());

                                (values.min(), values.max())
                            })
                            .evaluate()?;

                        let min = min.map(|min| min.1).unwrap_or(MedRecordValue::Null);
                        let max = max.map(|max| max.1).unwrap_or(MedRecordValue::Null);

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Temporal { min, max },
                        }
                    }
                    AttributeType::Unstructured => {
                        let values: Vec<_> = medrecord
                            .query_nodes(|nodes| {
                                node_selection_query(nodes);

                                nodes.attribute(key.clone())
                            })
                            .evaluate()?
                            .map(|(_, value)| value)
                            .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .dedup_by(|a, b| a == b)
                            .collect();

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Unstructured {
                                distinct_count: values.len(),
                            },
                        }
                    }
                };

                Ok::<_, MedRecordError>((key.clone(), attribute_overview))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self { count, attributes })
    }
}

#[derive(Debug, Clone)]
pub struct EdgeGroupOverview {
    pub count: usize,
    pub attributes: MrHashMap<MedRecordAttribute, AttributeOverview>,
}

impl Display for EdgeGroupOverview {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut builder = Builder::new();

        builder.push_record([
            "Edge Count",
            "Attribute",
            "Attribute Type",
            "Data Type",
            "Details",
        ]);

        for (attribute, overview) in &self.attributes {
            let details = overview.data.details();

            builder.push_record([
                &self.count.to_string(),
                &attribute.to_string(),
                overview.data.attribute_type_name(),
                &overview.data_type.to_string(),
                &details,
            ]);
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Edge Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        writeln!(f, "{table}")
    }
}

impl EdgeGroupOverview {
    fn new(
        medrecord: &MedRecord,
        group_schema: &GroupSchema,
        group: Option<&Group>,
    ) -> Result<Self, MedRecordError> {
        let edge_selection_query: Box<dyn Fn(&Wrapper<EdgeOperand>)> = match group {
            Some(group) => Box::new(|edges| {
                edges.in_group(group.clone());
            }),
            None => Box::new(|edges| {
                edges.exclude(|edges| {
                    let groups: Vec<_> = medrecord.groups().cloned().collect();

                    edges.in_group((groups, MatchMode::Any));
                });
            }),
        };

        let count = medrecord
            .query_edges(|edges| {
                edge_selection_query(edges);

                edges.index().count()
            })
            .evaluate()?
            .unwrap_or(0) as usize;

        let attributes: MrHashMap<_, _> = group_schema
            .edges()
            .iter()
            .map(|(key, attribute_data_type)| {
                let attribute_type = attribute_data_type.attribute_type();
                let data_type = attribute_data_type.data_type().clone();

                let attribute_overview = match attribute_type {
                    AttributeType::Categorical => {
                        let values = medrecord
                            .query_edges(|edges| {
                                edge_selection_query(edges);

                                edges.attribute(key.clone())
                            })
                            .evaluate()?
                            .map(|(_, value)| value)
                            .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .dedup_by(|a, b| a == b)
                            .collect();

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Categorical {
                                distinct_values: values,
                            },
                        }
                    }
                    AttributeType::Continuous => {
                        let (min, mean, max) = medrecord
                            .query_edges(|edges| {
                                edge_selection_query(edges);

                                let values = edges.attribute(key.clone());

                                (values.min(), values.mean(), values.max())
                            })
                            .evaluate()?;

                        let min = min.map(|min| min.1).unwrap_or(MedRecordValue::Null);
                        let mean = mean.unwrap_or(MedRecordValue::Null);
                        let max = max.map(|max| max.1).unwrap_or(MedRecordValue::Null);

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Continuous { min, mean, max },
                        }
                    }
                    AttributeType::Temporal => {
                        let (min, max) = medrecord
                            .query_edges(|edges| {
                                edge_selection_query(edges);

                                let values = edges.attribute(key.clone());

                                (values.min(), values.max())
                            })
                            .evaluate()?;

                        let min = min.map(|min| min.1).unwrap_or(MedRecordValue::Null);
                        let max = max.map(|max| max.1).unwrap_or(MedRecordValue::Null);

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Temporal { min, max },
                        }
                    }
                    AttributeType::Unstructured => {
                        let values: Vec<_> = medrecord
                            .query_edges(|edges| {
                                edge_selection_query(edges);

                                edges.attribute(key.clone())
                            })
                            .evaluate()?
                            .map(|(_, value)| value)
                            .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .dedup_by(|a, b| a == b)
                            .collect();

                        AttributeOverview {
                            data_type,
                            data: AttributeOverviewData::Unstructured {
                                distinct_count: values.len(),
                            },
                        }
                    }
                };

                Ok::<_, MedRecordError>((key.clone(), attribute_overview))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self { count, attributes })
    }
}

#[derive(Debug, Clone)]
pub struct GroupOverview {
    pub node_overview: NodeGroupOverview,
    pub edge_overview: EdgeGroupOverview,
}

impl GroupOverview {
    pub(crate) fn new(
        medrecord: &MedRecord,
        group: Option<&Group>,
    ) -> Result<Self, MedRecordError> {
        let schema = &medrecord.schema;

        let group_schema = match group {
            Some(g) => schema.group(g)?,
            None => schema.ungrouped(),
        };

        Ok(Self {
            node_overview: NodeGroupOverview::new(medrecord, group_schema, group)?,
            edge_overview: EdgeGroupOverview::new(medrecord, group_schema, group)?,
        })
    }
}

impl Display for GroupOverview {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.node_overview)?;
        writeln!(f, "{}", self.edge_overview)
    }
}

#[derive(Debug, Clone)]
pub struct Overview {
    pub ungrouped_overview: GroupOverview,
    pub grouped_overviews: MrHashMap<Group, GroupOverview>,
}

impl Display for Overview {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut builder = Builder::new();

        builder.push_record([
            "Group",
            "Node Count",
            "Attribute",
            "Attribute Type",
            "Data Type",
            "Details",
        ]);

        for (group, group_overview) in std::iter::once((None, &self.ungrouped_overview))
            .chain(self.grouped_overviews.iter().map(|(g, o)| (Some(g), o)))
        {
            for (attribute, overview) in &group_overview.node_overview.attributes {
                let group_name = group
                    .map(|g| g.to_string())
                    .unwrap_or_else(|| "Ungrouped".to_string());
                let details = overview.data.details();

                builder.push_record([
                    &group_name,
                    &group_overview.node_overview.count.to_string(),
                    &attribute.to_string(),
                    overview.data.attribute_type_name(),
                    &overview.data_type.to_string(),
                    &details,
                ]);
            }
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Node Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0, 1]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        writeln!(f, "{table}")?;

        let mut builder = Builder::new();

        builder.push_record([
            "Group",
            "Edge Count",
            "Type",
            "Attribute",
            "Data Type",
            "Details",
        ]);

        for (group, group_overview) in std::iter::once((None, &self.ungrouped_overview))
            .chain(self.grouped_overviews.iter().map(|(g, o)| (Some(g), o)))
        {
            for (attribute, overview) in &group_overview.edge_overview.attributes {
                let group_name = group
                    .map(|g| g.to_string())
                    .unwrap_or_else(|| "Ungrouped".to_string());
                let details = overview.data.details();

                builder.push_record([
                    &group_name,
                    &group_overview.edge_overview.count.to_string(),
                    &attribute.to_string(),
                    overview.data.attribute_type_name(),
                    &overview.data_type.to_string(),
                    &details,
                ]);
            }
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Edge Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0, 1]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        writeln!(f, "{table}")
    }
}

impl Overview {
    pub(crate) fn new(medrecord: &MedRecord) -> Result<Self, MedRecordError> {
        Ok(Self {
            ungrouped_overview: GroupOverview::new(medrecord, None)?,
            grouped_overviews: medrecord
                .groups()
                .map(|group| {
                    Ok::<_, MedRecordError>((
                        group.clone(),
                        GroupOverview::new(medrecord, Some(group))?,
                    ))
                })
                .collect::<Result<_, _>>()?,
        })
    }
}

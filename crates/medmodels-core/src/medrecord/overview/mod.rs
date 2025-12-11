#![allow(clippy::type_complexity)]

mod tabled_modifiers;

use crate::{
    errors::MedRecordError,
    medrecord::overview::tabled_modifiers::MergeDuplicatesVerticalByColumn,
    prelude::{AttributeType, DataType, Group, GroupSchema, MedRecordAttribute, MedRecordValue},
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{Display, Formatter},
};
use tabled::{
    builder::Builder,
    settings::{object::Columns, themes::BorderCorrection, Alignment, Panel, Style, Width},
};

pub const DEFAULT_TRUNCATE_DETAILS: usize = 80;

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

    truncate_details: Option<usize>,
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

        if self.attributes.is_empty() && self.count > 0 {
            builder.push_record([&self.count.to_string(), "-", "-", "-", "-"]);
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Node Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        if let Some(truncate_details) = self.truncate_details {
            table.modify(Columns::last(), Width::truncate(truncate_details));
        }

        writeln!(f, "{table}")
    }
}

impl NodeGroupOverview {
    fn new(
        medrecord: &MedRecord,
        group_schema: &GroupSchema,
        group: Option<&Group>,
        truncate_details: Option<usize>,
    ) -> Result<Self, MedRecordError> {
        let nodes_in_group: HashSet<_> = match group {
            Some(group) => medrecord.nodes_in_group(group)?.cloned().collect(),
            None => medrecord.ungrouped_nodes().cloned().collect(),
        };
        let count = nodes_in_group.len();

        let attributes: MrHashMap<_, _> = group_schema
            .nodes()
            .par_iter()
            .map(|(key, attribute_data_type)| {
                let attribute_type = attribute_data_type.attribute_type();
                let data_type = attribute_data_type.data_type().clone();

                let attribute_overview = match attribute_type {
                    AttributeType::Categorical => {
                        let values = medrecord
                            .query_nodes(|nodes| {
                                nodes.index().is_in(nodes_in_group.clone());

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
                                nodes.index().is_in(nodes_in_group.clone());

                                let values = nodes.attribute(key.clone());

                                values.exclude(|values| {
                                    values.is_null();
                                });

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
                                nodes.index().is_in(nodes_in_group.clone());

                                let values = nodes.attribute(key.clone());

                                values.exclude(|values| {
                                    values.is_null();
                                });

                                (values.min(), values.max())
                            })
                            .evaluate()
                            .unwrap();

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
                                nodes.index().is_in(nodes_in_group.clone());

                                nodes.attribute(key.clone())
                            })
                            .evaluate()
                            .unwrap()
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

        Ok(Self {
            count,
            attributes,
            truncate_details,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EdgeGroupOverview {
    pub count: usize,
    pub attributes: MrHashMap<MedRecordAttribute, AttributeOverview>,

    truncate_details: Option<usize>,
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

        if let Some(truncate_details) = self.truncate_details {
            table.modify(Columns::last(), Width::truncate(truncate_details));
        }

        writeln!(f, "{table}")
    }
}

impl EdgeGroupOverview {
    fn new(
        medrecord: &MedRecord,
        group_schema: &GroupSchema,
        group: Option<&Group>,
        truncate_details: Option<usize>,
    ) -> Result<Self, MedRecordError> {
        let edges_in_group: HashSet<_> = match group {
            Some(group) => medrecord.edges_in_group(group)?.cloned().collect(),
            None => medrecord.ungrouped_edges().cloned().collect(),
        };
        let count = edges_in_group.len();

        let attributes: MrHashMap<_, _> = group_schema
            .edges()
            .par_iter()
            .map(|(key, attribute_data_type)| {
                let attribute_type = attribute_data_type.attribute_type();
                let data_type = attribute_data_type.data_type().clone();

                let attribute_overview = match attribute_type {
                    AttributeType::Categorical => {
                        let values = medrecord
                            .query_edges(|edges| {
                                edges.index().is_in(edges_in_group.clone());

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
                                edges.index().is_in(edges_in_group.clone());

                                let values = edges.attribute(key.clone());

                                values.exclude(|values| {
                                    values.is_null();
                                });

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
                                edges.index().is_in(edges_in_group.clone());

                                let values = edges.attribute(key.clone());

                                values.exclude(|values| {
                                    values.is_null();
                                });

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
                                edges.index().is_in(edges_in_group.clone());

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

        Ok(Self {
            count,
            attributes,
            truncate_details,
        })
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
        truncate_details: Option<usize>,
    ) -> Result<Self, MedRecordError> {
        let schema = &medrecord.schema;

        let group_schema = match group {
            Some(g) => schema.group(g)?,
            None => schema.ungrouped(),
        };

        Ok(Self {
            node_overview: NodeGroupOverview::new(
                medrecord,
                group_schema,
                group,
                truncate_details,
            )?,
            edge_overview: EdgeGroupOverview::new(
                medrecord,
                group_schema,
                group,
                truncate_details,
            )?,
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

    truncate_details: Option<usize>,
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
            let group_name = group
                .map(|g| g.to_string())
                .unwrap_or_else(|| "Ungrouped".to_string());
            let count = group_overview.node_overview.count;

            for (attribute, overview) in &group_overview.node_overview.attributes {
                let details = overview.data.details();

                builder.push_record([
                    &group_name,
                    &count.to_string(),
                    &attribute.to_string(),
                    overview.data.attribute_type_name(),
                    &overview.data_type.to_string(),
                    &details,
                ]);
            }

            if group_overview.node_overview.attributes.is_empty() && count > 0 {
                builder.push_record([&group_name, &count.to_string(), "-", "-", "-", "-"]);
            }
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Node Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0, 1]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        if let Some(truncate_details) = self.truncate_details {
            table.modify(Columns::last(), Width::truncate(truncate_details));
        }

        writeln!(f, "{table}")?;

        let mut builder = Builder::new();

        builder.push_record([
            "Group",
            "Edge Count",
            "Attribute",
            "Attribute Type",
            "Data Type",
            "Details",
        ]);

        for (group, group_overview) in std::iter::once((None, &self.ungrouped_overview))
            .chain(self.grouped_overviews.iter().map(|(g, o)| (Some(g), o)))
        {
            let group_name = group
                .map(|g| g.to_string())
                .unwrap_or_else(|| "Ungrouped".to_string());
            let count = group_overview.edge_overview.count;

            for (attribute, overview) in &group_overview.edge_overview.attributes {
                let details = overview.data.details();

                builder.push_record([
                    &group_name,
                    &count.to_string(),
                    &attribute.to_string(),
                    overview.data.attribute_type_name(),
                    &overview.data_type.to_string(),
                    &details,
                ]);
            }

            if group_overview.edge_overview.attributes.is_empty() && count > 0 {
                builder.push_record([&group_name, &count.to_string(), "-", "-", "-", "-"]);
            }
        }

        let mut table = builder.build();
        table.with(Style::modern());
        table.with(Panel::header("Edge Overview"));
        table.with(MergeDuplicatesVerticalByColumn::new(vec![0, 1]));
        table.with(Alignment::center_vertical());
        table.with(BorderCorrection {});

        if let Some(truncate_details) = self.truncate_details {
            table.modify(Columns::last(), Width::truncate(truncate_details));
        }

        writeln!(f, "{table}")
    }
}

impl Overview {
    pub(crate) fn new(
        medrecord: &MedRecord,
        truncate_details: Option<usize>,
    ) -> Result<Self, MedRecordError> {
        Ok(Self {
            ungrouped_overview: GroupOverview::new(medrecord, None, truncate_details)?,
            grouped_overviews: medrecord
                .groups()
                .map(|group| {
                    Ok::<_, MedRecordError>((
                        group.clone(),
                        GroupOverview::new(medrecord, Some(group), truncate_details)?,
                    ))
                })
                .collect::<Result<_, _>>()?,
            truncate_details,
        })
    }
}

use super::{
    operand::{
        EdgeIndexComparisonOperand, EdgeIndexOperand, EdgeIndicesComparisonOperand,
        EdgeIndicesOperand,
    },
    BinaryArithmeticKind, EdgeOperand, MultipleComparisonKind, SingleComparisonKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{Contains, EndsWith, Mod, StartsWith},
        querying::{
            attributes::AttributesTreeOperand,
            edges::SingleKind,
            group_by::{GroupOperand, PartitionGroups},
            nodes::NodeOperand,
            tee_grouped_iterator,
            values::{MultipleValuesWithIndexContext, MultipleValuesWithIndexOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator, DeepClone, EvaluateForward, EvaluateForwardGrouped, GroupedIterator,
            ReadWriteOrPanic,
        },
        EdgeIndex, Group, MedRecordAttribute, MedRecordValue,
    },
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashSet;
use rand::{rng, seq::IteratorRandom};
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Values {
        operand: Wrapper<MultipleValuesWithIndexOperand<EdgeOperand>>,
    },
    Attributes {
        operand: Wrapper<AttributesTreeOperand<EdgeOperand>>,
    },
    Indices {
        operand: Wrapper<EdgeIndicesOperand>,
    },

    InGroup {
        group: CardinalityWrapper<Group>,
    },
    HasAttribute {
        attribute: CardinalityWrapper<MedRecordAttribute>,
    },

    SourceNode {
        operand: Wrapper<NodeOperand>,
    },
    TargetNode {
        operand: Wrapper<NodeOperand>,
    },

    EitherOr {
        either: Wrapper<EdgeOperand>,
        or: Wrapper<EdgeOperand>,
    },
    Exclude {
        operand: Wrapper<EdgeOperand>,
    },

    GroupBy {
        operand: Wrapper<GroupOperand<EdgeOperand>>,
    },
}

impl DeepClone for EdgeOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Values { operand } => Self::Values {
                operand: operand.deep_clone(),
            },
            Self::Attributes { operand } => Self::Attributes {
                operand: operand.deep_clone(),
            },
            Self::Indices { operand } => Self::Indices {
                operand: operand.deep_clone(),
            },
            Self::InGroup { group } => Self::InGroup {
                group: group.clone(),
            },
            Self::HasAttribute { attribute } => Self::HasAttribute {
                attribute: attribute.clone(),
            },
            Self::SourceNode { operand } => Self::SourceNode {
                operand: operand.deep_clone(),
            },
            Self::TargetNode { operand } => Self::TargetNode {
                operand: operand.deep_clone(),
            },
            Self::EitherOr { either, or } => Self::EitherOr {
                either: either.deep_clone(),
                or: or.deep_clone(),
            },
            Self::Exclude { operand } => Self::Exclude {
                operand: operand.deep_clone(),
            },
            Self::GroupBy { operand } => Self::GroupBy {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl EdgeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, &'a EdgeIndex>> {
        Ok(match self {
            Self::Values { operand } => {
                Box::new(Self::evaluate_values(medrecord, edge_indices, operand)?)
            }
            Self::Attributes { operand } => {
                Box::new(Self::evaluate_attributes(medrecord, edge_indices, operand)?)
            }
            Self::Indices { operand } => {
                Box::new(Self::evaluate_indices(medrecord, edge_indices, operand)?)
            }
            Self::InGroup { group } => Box::new(Self::evaluate_in_group(
                medrecord,
                edge_indices,
                group.clone(),
            )),
            Self::HasAttribute { attribute } => Box::new(Self::evaluate_has_attribute(
                medrecord,
                edge_indices,
                attribute.clone(),
            )),
            Self::SourceNode { operand } => Box::new(Self::evaluate_source_node(
                medrecord,
                edge_indices,
                operand,
            )?),
            Self::TargetNode { operand } => Box::new(Self::evaluate_target_node(
                medrecord,
                edge_indices,
                operand,
            )?),
            Self::EitherOr { either, or } => {
                let (edge_indices_1, rest) = edge_indices.tee();
                let (edge_indices_2, edge_indices_3) = rest.tee();

                let either_set: MrHashSet<_> = either
                    .evaluate_forward(medrecord, Box::new(edge_indices_1))?
                    .collect();
                let or_set: MrHashSet<_> = or
                    .evaluate_forward(medrecord, Box::new(edge_indices_2))?
                    .collect();

                Box::new(
                    edge_indices_3
                        .filter(move |edge| either_set.contains(edge) || or_set.contains(edge)),
                )
            }
            Self::Exclude { operand } => {
                let (edge_indices_1, edge_indices_2) = edge_indices.tee();

                let result: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(edge_indices_1))?
                    .collect();

                Box::new(edge_indices_2.filter(move |node_index| !result.contains(node_index)))
            }
            Self::GroupBy { operand } => {
                Box::new(Self::evaluate_group_by(medrecord, edge_indices, operand)?)
            }
        })
    }

    #[inline]
    pub(crate) fn get_values<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = (&'a EdgeIndex, MedRecordValue)> {
        edge_indices.flat_map(move |edge_index| {
            Some((
                edge_index,
                medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        })
    }

    #[inline]
    fn evaluate_values<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: &Wrapper<MultipleValuesWithIndexOperand<EdgeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let MultipleValuesWithIndexContext::Operand((_, ref attribute)) =
            operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let values = Self::get_values(medrecord, edge_indices, attribute.clone());

        Ok(operand
            .evaluate_forward(medrecord, Box::new(values))?
            .map(|value| value.0))
    }

    #[inline]
    pub(crate) fn get_attributes<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
    ) -> impl Iterator<Item = (&'a EdgeIndex, Vec<MedRecordAttribute>)> {
        edge_indices.map(move |edge_index| {
            let attributes = medrecord
                .edge_attributes(edge_index)
                .expect("Edge must exist")
                .keys()
                .cloned();

            (edge_index, attributes.collect())
        })
    }

    #[inline]
    fn evaluate_attributes<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: &Wrapper<AttributesTreeOperand<EdgeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let attributes = Self::get_attributes(medrecord, edge_indices);

        Ok(operand
            .evaluate_forward(medrecord, Box::new(attributes))?
            .map(|value| value.0))
    }

    #[inline]
    fn evaluate_indices<'a>(
        medrecord: &MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(edge_indices_1.cloned()))?
            .collect();

        Ok(edge_indices_2
            .into_iter()
            .filter(move |index| result.contains(*index)))
    }

    #[inline]
    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        group: CardinalityWrapper<Group>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |edge_index| {
            let groups_of_edge = medrecord
                .groups_of_edge(edge_index)
                .expect("Node must exist");

            let groups_of_edge: MrHashSet<_> = groups_of_edge.collect();

            match &group {
                CardinalityWrapper::Single(group) => groups_of_edge.contains(&group),
                CardinalityWrapper::Multiple(groups) => {
                    groups.iter().all(|group| groups_of_edge.contains(&group))
                }
            }
        })
    }

    #[inline]
    fn evaluate_has_attribute<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        attribute: CardinalityWrapper<MedRecordAttribute>,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        edge_indices.filter(move |edge_index| {
            let attributes_of_edge = medrecord
                .edge_attributes(edge_index)
                .expect("Node must exist")
                .keys();

            let attributes_of_edge: MrHashSet<_> = attributes_of_edge.collect();

            match &attribute {
                CardinalityWrapper::Single(attribute) => attributes_of_edge.contains(&attribute),
                CardinalityWrapper::Multiple(attributes) => attributes
                    .iter()
                    .all(|attribute| attributes_of_edge.contains(&attribute)),
            }
        })
    }

    #[inline]
    fn evaluate_source_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

        let node_indices = edge_indices_1.map(|edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            edge_endpoints.0
        });

        let node_indices: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(node_indices))?
            .collect();

        Ok(edge_indices_2.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.0)
        }))
    }

    #[inline]
    fn evaluate_target_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

        let node_indices = edge_indices_1.map(|edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            edge_endpoints.1
        });

        let node_indices: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(node_indices))?
            .collect();

        Ok(edge_indices_2.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        }))
    }

    fn evaluate_group_by<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
        operand: &Wrapper<GroupOperand<EdgeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        Ok(EdgeOperand::merge(
            operand.evaluate_forward(medrecord, Box::new(edge_indices))?,
        ))
    }
}

impl EdgeOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        Ok(match self {
            EdgeOperation::Values { operand } => Box::new(Self::evaluate_values_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::Attributes { operand } => Box::new(Self::evaluate_attributes_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::Indices { operand } => Box::new(Self::evaluate_indices_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::InGroup { group } => {
                let group = group.clone();

                Box::new(edge_indices.map(move |(key, edge_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_in_group(
                            medrecord,
                            edge_indices,
                            group.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            EdgeOperation::HasAttribute { attribute } => {
                let attribute = attribute.clone();

                Box::new(edge_indices.map(move |(key, edge_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_has_attribute(
                            medrecord,
                            edge_indices,
                            attribute.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            EdgeOperation::SourceNode { operand } => Box::new(Self::evaluate_source_node_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::TargetNode { operand } => Box::new(Self::evaluate_target_node_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::EitherOr { either, or } => Box::new(Self::evaluate_either_or_grouped(
                medrecord,
                edge_indices,
                either,
                or,
            )?),
            EdgeOperation::Exclude { operand } => Box::new(Self::evaluate_exclude_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeOperation::GroupBy { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    fn evaluate_values_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<MultipleValuesWithIndexOperand<EdgeOperand>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let MultipleValuesWithIndexContext::Operand((_, ref attribute)) =
            operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let values: Vec<_> = edge_indices
            .map(|(key, edge_indices)| {
                (
                        key,
                        Box::new(Self::get_values(medrecord, edge_indices, attribute.clone()))
                            as <MultipleValuesWithIndexOperand<EdgeOperand> as EvaluateForward<
                                'a,
                            >>::InputValue,
                    )
            })
            .collect();

        Ok(Box::new(
            operand
                .evaluate_forward_grouped(medrecord, Box::new(values.into_iter()))?
                .map(|(key, values)| {
                    (
                        key,
                        Box::new(values.map(|value| value.0)) as BoxedIterator<_>,
                    )
                }),
        ))
    }

    #[inline]
    fn evaluate_attributes_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<AttributesTreeOperand<EdgeOperand>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let attributes = edge_indices.map(|(key, edge_indices)| {
            (
                key,
                Box::new(Self::get_attributes(medrecord, edge_indices))
                    as <AttributesTreeOperand<EdgeOperand> as EvaluateForward<'a>>::InputValue,
            )
        });

        Ok(Box::new(
            operand
                .evaluate_forward_grouped(medrecord, Box::new(attributes))?
                .map(|(key, attributes)| {
                    (
                        key,
                        Box::new(attributes.map(|value| value.0)) as BoxedIterator<_>,
                    )
                }),
        ))
    }

    #[inline]
    fn evaluate_indices_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let edge_indices_1 = edge_indices_1
            .map(|(key, edge_indices)| (key, Box::new(edge_indices.cloned()) as BoxedIterator<_>));

        let mut edge_indices_1: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(edge_indices_1))?
            .collect();

        Ok(Box::new(edge_indices_2.map(move |(key, edge_indices)| {
            let edge_indices_position = &edge_indices_1
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let edge_indices_1: MrHashSet<_> =
                edge_indices_1.remove(*edge_indices_position).1.collect();

            let filtered_indices: Vec<_> = edge_indices
                .filter(|edge_index| edge_indices_1.contains(*edge_index))
                .collect();

            (
                key,
                Box::new(filtered_indices.into_iter()) as BoxedIterator<_>,
            )
        })))
    }

    #[inline]
    fn evaluate_source_node_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let node_indices = edge_indices_1.map(|(key, edge_indices)| {
            let node_indices: BoxedIterator<_> = Box::new(edge_indices.map(|edge_index| {
                let edge_endpoints = medrecord
                    .edge_endpoints(edge_index)
                    .expect("Edge must exist");

                edge_endpoints.0
            }));

            (key, node_indices)
        });

        let mut node_indices: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(node_indices))?
            .collect();

        Ok(Box::new(edge_indices_2.map(move |(key, edge_indices)| {
            let node_indices_position = &node_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let node_indices: MrHashSet<_> =
                node_indices.remove(*node_indices_position).1.collect();

            let filtered_indices: Vec<_> = edge_indices
                .filter(|edge_index| {
                    let edge_endpoints = medrecord
                        .edge_endpoints(edge_index)
                        .expect("Edge must exist");

                    node_indices.contains(edge_endpoints.0)
                })
                .collect();

            (
                key,
                Box::new(filtered_indices.into_iter()) as BoxedIterator<_>,
            )
        })))
    }

    #[inline]
    fn evaluate_target_node_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let node_indices = edge_indices_1.map(|(key, edge_indices)| {
            let node_indices: BoxedIterator<_> = Box::new(edge_indices.map(|edge_index| {
                let edge_endpoints = medrecord
                    .edge_endpoints(edge_index)
                    .expect("Edge must exist");

                edge_endpoints.1
            }));

            (key, node_indices)
        });

        let mut node_indices: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(node_indices))?
            .collect();

        Ok(Box::new(edge_indices_2.map(move |(key, edge_indices)| {
            let node_indices_position = &node_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let node_indices: MrHashSet<_> =
                node_indices.remove(*node_indices_position).1.collect();

            let filtered_indices: Vec<_> = edge_indices
                .filter(|edge_index| {
                    let edge_endpoints = medrecord
                        .edge_endpoints(edge_index)
                        .expect("Edge must exist");

                    node_indices.contains(edge_endpoints.1)
                })
                .collect();

            (
                key,
                Box::new(filtered_indices.into_iter()) as BoxedIterator<_>,
            )
        })))
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        either: &Wrapper<EdgeOperand>,
        or: &Wrapper<EdgeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let either_indices = either.evaluate_forward_grouped(medrecord, edge_indices_1)?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, edge_indices_2)?
            .collect();

        let edge_indices = either_indices.map(move |(key, either_indices)| {
            let indices_position = or_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_indices = or_indices.remove(indices_position).1;

            let edge_indices: BoxedIterator<_> = Box::new(
                either_indices
                    .chain(or_indices)
                    .unique_by(|edge_index| **edge_index),
            );

            (key, edge_indices)
        });

        Ok(Box::new(edge_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>,
        operand: &Wrapper<EdgeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, edge_indices_1)?
            .collect();

        let edge_indices = edge_indices_2.map(move |(key, values)| {
            let indices_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_indices: MrHashSet<_> = result.remove(indices_position).1.collect();

            let edge_indices: BoxedIterator<_> =
                Box::new(values.filter(move |edge_index| !excluded_indices.contains(edge_index)));

            (key, edge_indices)
        });

        Ok(Box::new(edge_indices))
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndicesOperation {
    EdgeIndexOperation {
        operand: Wrapper<EdgeIndexOperand>,
    },
    EdgeIndexComparisonOperation {
        operand: EdgeIndexComparisonOperand,
        kind: SingleComparisonKind,
    },
    EdgeIndicesComparisonOperation {
        operand: EdgeIndicesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: EdgeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    },

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<EdgeIndicesOperand>,
        or: Wrapper<EdgeIndicesOperand>,
    },
    Exclude {
        operand: Wrapper<EdgeIndicesOperand>,
    },

    Merge {
        operand: Wrapper<EdgeIndicesOperand>,
    },
}

impl DeepClone for EdgeIndicesOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::EdgeIndexOperation { operand } => Self::EdgeIndexOperation {
                operand: operand.deep_clone(),
            },
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::EdgeIndexComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::EdgeIndicesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::IsMax => Self::IsMax,
            Self::IsMin => Self::IsMin,
            Self::EitherOr { either, or } => Self::EitherOr {
                either: either.deep_clone(),
                or: or.deep_clone(),
            },
            Self::Exclude { operand } => Self::Exclude {
                operand: operand.deep_clone(),
            },
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl EdgeIndicesOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        Ok(match self {
            Self::EdgeIndexOperation { operand } => {
                Self::evaluate_edge_index_operation(medrecord, indices, operand)?
            }
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_edge_index_comparison_operation(medrecord, indices, operand, kind)?
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_edge_indices_comparison_operation(medrecord, indices, operand, kind)?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, indices, operand, kind)?,
            ),
            Self::IsMax => Self::evaluate_is_max(indices),
            Self::IsMin => Self::evaluate_is_min(indices),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, indices, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, indices, operand)?,
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    pub(crate) fn get_max(indices: impl Iterator<Item = EdgeIndex>) -> Option<EdgeIndex> {
        indices.max()
    }

    #[inline]
    pub(crate) fn get_min(indices: impl Iterator<Item = EdgeIndex>) -> Option<EdgeIndex> {
        indices.min()
    }
    #[inline]
    pub(crate) fn get_count(indices: impl Iterator<Item = EdgeIndex>) -> EdgeIndex {
        indices.count() as EdgeIndex
    }

    #[inline]
    pub(crate) fn get_sum(indices: impl Iterator<Item = EdgeIndex>) -> EdgeIndex {
        indices.sum()
    }

    #[inline]
    pub(crate) fn get_random(indices: impl Iterator<Item = EdgeIndex>) -> Option<EdgeIndex> {
        indices.choose(&mut rng())
    }

    #[inline]
    fn evaluate_edge_index_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        operand: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let kind = &operand.0.read_or_panic().kind;

        let index = match kind {
            SingleKind::Max => EdgeIndicesOperation::get_max(indices_1),
            SingleKind::Min => EdgeIndicesOperation::get_min(indices_1),
            SingleKind::Count => Some(EdgeIndicesOperation::get_count(indices_1)),
            SingleKind::Sum => Some(EdgeIndicesOperation::get_sum(indices_1)),
            SingleKind::Random => EdgeIndicesOperation::get_random(indices_1),
        };

        Ok(match operand.evaluate_forward(medrecord, index)? {
            Some(_) => Box::new(indices_2.into_iter()),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_edge_index_comparison_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        comparison_operand: &EdgeIndexComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let comparison_index =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => Ok(Box::new(
                indices.filter(move |index| index > &comparison_index),
            )),
            SingleComparisonKind::GreaterThanOrEqualTo => Ok(Box::new(
                indices.filter(move |index| index >= &comparison_index),
            )),
            SingleComparisonKind::LessThan => Ok(Box::new(
                indices.filter(move |index| index < &comparison_index),
            )),
            SingleComparisonKind::LessThanOrEqualTo => Ok(Box::new(
                indices.filter(move |index| index <= &comparison_index),
            )),
            SingleComparisonKind::EqualTo => Ok(Box::new(
                indices.filter(move |index| index == &comparison_index),
            )),
            SingleComparisonKind::NotEqualTo => Ok(Box::new(
                indices.filter(move |index| index != &comparison_index),
            )),
            SingleComparisonKind::StartsWith => Ok(Box::new(
                indices.filter(move |index| index.starts_with(&comparison_index)),
            )),
            SingleComparisonKind::EndsWith => Ok(Box::new(
                indices.filter(move |index| index.ends_with(&comparison_index)),
            )),
            SingleComparisonKind::Contains => Ok(Box::new(
                indices.filter(move |index| index.contains(&comparison_index)),
            )),
        }
    }

    #[inline]
    fn evaluate_edge_indices_comparison_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        comparison_operand: &EdgeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let comparison_indices = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => Ok(Box::new(
                indices.filter(move |index| comparison_indices.contains(index)),
            )),
            MultipleComparisonKind::IsNotIn => Ok(Box::new(
                indices.filter(move |index| !comparison_indices.contains(index)),
            )),
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = EdgeIndex>,
        operand: &EdgeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = EdgeIndex>> {
        let arithmetic_index =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        Ok(indices
            .map(move |index| match kind {
                BinaryArithmeticKind::Add => Ok(index.add(arithmetic_index)),
                BinaryArithmeticKind::Sub => Ok(index.sub(arithmetic_index)),
                BinaryArithmeticKind::Mul => Ok(index.mul(arithmetic_index)),
                BinaryArithmeticKind::Pow => Ok(index.pow(arithmetic_index)),
                BinaryArithmeticKind::Mod => index.r#mod(arithmetic_index),
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    fn evaluate_is_max<'a>(
        indices: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> BoxedIterator<'a, EdgeIndex> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let max_index = Self::get_max(indices_1);

        let Some(max_index) = max_index else {
            return Box::new(std::iter::empty());
        };

        Box::new(indices_2.filter(move |index| *index == max_index))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        indices: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> BoxedIterator<'a, EdgeIndex> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let min_index = Self::get_min(indices_1);

        let Some(min_index) = min_index else {
            return Box::new(std::iter::empty());
        };

        Box::new(indices_2.filter(move |index| *index == min_index))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        either: &Wrapper<EdgeIndicesOperand>,
        or: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let either_indices = either.evaluate_forward(medrecord, Box::new(indices_1))?;
        let or_indices = or.evaluate_forward(medrecord, Box::new(indices_2))?;

        Ok(Box::new(either_indices.chain(or_indices).unique()))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        operand: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(indices_1))?
            .collect();

        Ok(Box::new(
            indices_2.filter(move |index| !result.contains(index)),
        ))
    }
}

impl EdgeIndicesOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>> {
        Ok(match self {
            EdgeIndicesOperation::EdgeIndexOperation { operand } => Box::new(
                Self::evaluate_edge_index_operation_grouped(medrecord, edge_indices, operand)?,
            ),
            EdgeIndicesOperation::EdgeIndexComparisonOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_edge_index_comparison_operation(
                                medrecord,
                                edge_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndicesOperation::EdgeIndicesComparisonOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_edge_indices_comparison_operation(
                                medrecord,
                                edge_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndicesOperation::BinaryArithmeticOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord,
                                edge_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndicesOperation::IsMax => Box::new(
                edge_indices
                    .map(move |(key, edge_indices)| (key, Self::evaluate_is_max(edge_indices))),
            ),
            EdgeIndicesOperation::IsMin => Box::new(
                edge_indices
                    .map(move |(key, edge_indices)| (key, Self::evaluate_is_min(edge_indices))),
            ),
            EdgeIndicesOperation::EitherOr { either, or } => Box::new(
                Self::evaluate_either_or_grouped(medrecord, edge_indices, either, or)?,
            ),
            EdgeIndicesOperation::Exclude { operand } => Box::new(Self::evaluate_exclude_grouped(
                medrecord,
                edge_indices,
                operand,
            )?),
            EdgeIndicesOperation::Merge { operand } => {
                let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

                let edge_indices_1 = edge_indices_1.flat_map(|(_, value)| value);

                let edge_indinces_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(edge_indices_1))?
                    .collect();

                Box::new(edge_indices_2.map(move |(key, edge_indices)| {
                    let edge_indices: Vec<_> = edge_indices
                        .filter(|edge_index| edge_indinces_1.contains(edge_index))
                        .collect();

                    let edge_indices: BoxedIterator<_> = Box::new(edge_indices.into_iter());

                    (key, edge_indices)
                }))
            }
        })
    }

    #[inline]
    fn evaluate_edge_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>,
        operand: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);
        let mut edge_indices_2: Vec<_> = edge_indices_2.collect();

        let kind = &operand.0.read_or_panic().kind;

        let edge_indices_1: Vec<_> = edge_indices_1
            .map(move |(key, edge_indices)| {
                Ok((
                    key,
                    match kind {
                        SingleKind::Max => EdgeIndicesOperation::get_max(edge_indices),
                        SingleKind::Min => EdgeIndicesOperation::get_min(edge_indices),
                        SingleKind::Count => Some(EdgeIndicesOperation::get_count(edge_indices)),
                        SingleKind::Sum => Some(EdgeIndicesOperation::get_sum(edge_indices)),
                        SingleKind::Random => EdgeIndicesOperation::get_random(edge_indices),
                    },
                ))
            })
            .collect::<MedRecordResult<_>>()?;

        let edge_indices_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(edge_indices_1.into_iter()))?;

        Ok(Box::new(edge_indices_1.map(
            move |(key, value)| match value {
                Some(_) => {
                    let edge_indices_position = edge_indices_2
                        .iter()
                        .position(|(k, _)| k == &key)
                        .expect("Entry must exist");

                    edge_indices_2.remove(edge_indices_position)
                }
                None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
            },
        )))
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>,
        either: &Wrapper<EdgeIndicesOperand>,
        or: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let either_indices = either.evaluate_forward_grouped(medrecord, edge_indices_1)?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, edge_indices_2)?
            .collect();

        let edge_indices = either_indices.map(move |(key, either_indices)| {
            let indices_position = or_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_indices = or_indices.remove(indices_position).1;

            let edge_indices: BoxedIterator<_> = Box::new(
                either_indices
                    .chain(or_indices)
                    .unique_by(|edge_index| *edge_index),
            );

            (key, edge_indices)
        });

        Ok(Box::new(edge_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>,
        operand: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = tee_grouped_iterator(edge_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, edge_indices_1)?
            .collect();

        let edge_indices = edge_indices_2.map(move |(key, values)| {
            let indices_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_indices: MrHashSet<_> = result.remove(indices_position).1.collect();

            let edge_indices: BoxedIterator<_> =
                Box::new(values.filter(move |edge_index| !excluded_indices.contains(edge_index)));

            (key, edge_indices)
        });

        Ok(Box::new(edge_indices))
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndexOperation {
    EdgeIndexComparisonOperation {
        operand: EdgeIndexComparisonOperand,
        kind: SingleComparisonKind,
    },
    EdgeIndicesComparisonOperation {
        operand: EdgeIndicesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: EdgeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    },

    EitherOr {
        either: Wrapper<EdgeIndexOperand>,
        or: Wrapper<EdgeIndexOperand>,
    },
    Exclude {
        operand: Wrapper<EdgeIndexOperand>,
    },

    Merge {
        operand: Wrapper<EdgeIndicesOperand>,
    },
}

impl DeepClone for EdgeIndexOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::EdgeIndexComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::EdgeIndicesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::EitherOr { either, or } => Self::EitherOr {
                either: either.deep_clone(),
                or: or.deep_clone(),
            },
            Self::Exclude { operand } => Self::Exclude {
                operand: operand.deep_clone(),
            },
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl EdgeIndexOperation {
    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        index: Option<EdgeIndex>,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let Some(index) = index else {
            return Ok(None);
        };

        match self {
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_edge_index_comparison_operation(medrecord, index, operand, kind)
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_edge_indcies_comparison_operation(medrecord, index, operand, kind)
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, index, operand, kind)
            }
            Self::EitherOr { either, or } => Self::evaluate_either_or(medrecord, index, either, or),
            Self::Exclude { operand } => {
                let result = operand.evaluate_forward(medrecord, Some(index))?.is_some();

                Ok(if result { None } else { Some(index) })
            }
            Self::Merge { operand: _ } => unreachable!(),
        }
    }

    #[inline]
    fn evaluate_edge_index_comparison_operation(
        medrecord: &MedRecord,
        index: EdgeIndex,
        comparison_operand: &EdgeIndexComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let comparison_index =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => index > comparison_index,
            SingleComparisonKind::GreaterThanOrEqualTo => index >= comparison_index,
            SingleComparisonKind::LessThan => index < comparison_index,
            SingleComparisonKind::LessThanOrEqualTo => index <= comparison_index,
            SingleComparisonKind::EqualTo => index == comparison_index,
            SingleComparisonKind::NotEqualTo => index != comparison_index,
            SingleComparisonKind::StartsWith => index.starts_with(&comparison_index),
            SingleComparisonKind::EndsWith => index.ends_with(&comparison_index),
            SingleComparisonKind::Contains => index.contains(&comparison_index),
        };

        Ok(if comparison_result { Some(index) } else { None })
    }

    #[inline]
    fn evaluate_edge_indcies_comparison_operation(
        medrecord: &MedRecord,
        index: EdgeIndex,
        comparison_operand: &EdgeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let comparison_indices = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_indices.contains(&index),
            MultipleComparisonKind::IsNotIn => !comparison_indices.contains(&index),
        };

        Ok(if comparison_result { Some(index) } else { None })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        index: EdgeIndex,
        operand: &EdgeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let arithmetic_index =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => index.add(arithmetic_index),
            BinaryArithmeticKind::Sub => index.sub(arithmetic_index),
            BinaryArithmeticKind::Mul => index.mul(arithmetic_index),
            BinaryArithmeticKind::Pow => index.pow(arithmetic_index),
            BinaryArithmeticKind::Mod => index.r#mod(arithmetic_index)?,
        }))
    }

    #[inline]
    fn evaluate_either_or(
        medrecord: &MedRecord,
        index: EdgeIndex,
        either: &Wrapper<EdgeIndexOperand>,
        or: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let either_result = either.evaluate_forward(medrecord, Some(index))?;
        let or_result = or.evaluate_forward(medrecord, Some(index))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl EdgeIndexOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, Option<EdgeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<EdgeIndex>>> {
        Ok(match self {
            EdgeIndexOperation::EdgeIndexComparisonOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_index)| {
                        let Some(edge_index) = edge_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_edge_index_comparison_operation(
                                medrecord, edge_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndexOperation::EdgeIndicesComparisonOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_index)| {
                        let Some(edge_index) = edge_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_edge_indcies_comparison_operation(
                                medrecord, edge_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndexOperation::BinaryArithmeticOperation { operand, kind } => Box::new(
                edge_indices
                    .map(move |(key, edge_index)| {
                        let Some(edge_index) = edge_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, edge_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            EdgeIndexOperation::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, edge_indices, either, or)?
            }
            EdgeIndexOperation::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, edge_indices, operand)?
            }
            EdgeIndexOperation::Merge { operand } => {
                let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

                let edge_indices_1 = edge_indices_1.filter_map(|(_, indices)| indices);

                let edge_indices_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(edge_indices_1))?
                    .collect();

                Box::new(edge_indices_2.map(move |(key, edge_index)| {
                    let edge_index = edge_index.filter(|index| edge_indices_1.contains(index));

                    (key, edge_index)
                }))
            }
        })
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, Option<EdgeIndex>>,
        either: &Wrapper<EdgeIndexOperand>,
        or: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

        let either_indices =
            either.evaluate_forward_grouped(medrecord, Box::new(edge_indices_1))?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(edge_indices_2))?
            .collect();

        let edge_indices = either_indices.map(move |(key, either_indices)| {
            let indices_position = or_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_index = or_indices.remove(indices_position).1;

            let index = match (either_indices, or_index) {
                (Some(either_result), _) => Some(either_result),
                (None, Some(or_result)) => Some(or_result),
                _ => None,
            };

            (key, index)
        });

        Ok(Box::new(edge_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, Option<EdgeIndex>>,
        operand: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<EdgeIndex>>> {
        let (edge_indices_1, edge_indices_2) = Itertools::tee(edge_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(edge_indices_1))?
            .collect();

        let edge_indices = edge_indices_2.map(move |(key, edge_index)| {
            let index_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_index = result.remove(index_position).1;

            let edge_index = match excluded_index {
                Some(_) => None,
                None => edge_index,
            };

            (key, edge_index)
        });

        Ok(Box::new(edge_indices))
    }
}

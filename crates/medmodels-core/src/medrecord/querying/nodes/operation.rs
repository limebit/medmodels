use super::{
    operand::{
        NodeIndexComparisonOperand, NodeIndexOperand, NodeIndicesComparisonOperand,
        NodeIndicesOperand,
    },
    BinaryArithmeticKind, MultipleComparisonKind, NodeOperand, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Contains, DataType, EndsWith, Lowercase, Mod, Pow, Slice, StartsWith, Trim,
            TrimEnd, TrimStart, Uppercase,
        },
        querying::{
            attributes::AttributesTreeOperand,
            edges::EdgeOperand,
            group_by::{GroupOperand, PartitionGroups},
            tee_grouped_iterator,
            values::{MultipleValuesWithIndexContext, MultipleValuesWithIndexOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator, DeepClone, EvaluateForward, EvaluateForwardGrouped, GroupedIterator,
            ReadWriteOrPanic,
        },
        Group, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex,
    },
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashSet;
use rand::{rng, seq::IteratorRandom};
use roaring::RoaringBitmap;
use std::{
    cmp::Ordering,
    collections::HashSet,
    ops::{Add, Mul, Range, Sub},
};

#[derive(Debug, Clone)]
pub enum EdgeDirection {
    Incoming,
    Outgoing,
    Both,
}

#[derive(Debug, Clone)]
pub enum NodeOperation {
    Values {
        operand: Wrapper<MultipleValuesWithIndexOperand<NodeOperand>>,
    },
    Attributes {
        operand: Wrapper<AttributesTreeOperand<NodeOperand>>,
    },
    Indices {
        operand: Wrapper<NodeIndicesOperand>,
    },

    InGroup {
        group: CardinalityWrapper<Group>,
    },
    HasAttribute {
        attribute: CardinalityWrapper<MedRecordAttribute>,
    },

    Edges {
        operand: Wrapper<EdgeOperand>,
        direction: EdgeDirection,
    },

    Neighbors {
        operand: Wrapper<NodeOperand>,
        direction: EdgeDirection,
    },

    EitherOr {
        either: Wrapper<NodeOperand>,
        or: Wrapper<NodeOperand>,
    },
    Exclude {
        operand: Wrapper<NodeOperand>,
    },

    GroupBy {
        operand: Wrapper<GroupOperand<NodeOperand>>,
    },
}

impl DeepClone for NodeOperation {
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
            Self::Edges { operand, direction } => Self::Edges {
                operand: operand.deep_clone(),
                direction: direction.clone(),
            },
            Self::Neighbors {
                operand,
                direction: drection,
            } => Self::Neighbors {
                operand: operand.deep_clone(),
                direction: drection.clone(),
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

impl NodeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
        Ok(match self {
            Self::Values { operand } => {
                Box::new(Self::evaluate_values(medrecord, node_indices, operand)?)
            }
            Self::Attributes { operand } => {
                Box::new(Self::evaluate_attributes(medrecord, node_indices, operand)?)
            }
            Self::Indices { operand } => {
                Box::new(Self::evaluate_indices(medrecord, node_indices, operand)?)
            }
            Self::InGroup { group } => Box::new(Self::evaluate_in_group(
                medrecord,
                node_indices,
                group.clone(),
            )),
            Self::HasAttribute { attribute } => Box::new(Self::evaluate_has_attribute(
                medrecord,
                node_indices,
                attribute.clone(),
            )),
            Self::Edges { operand, direction } => Box::new(Self::evaluate_edges(
                medrecord,
                node_indices,
                operand,
                direction.clone(),
            )?),
            Self::Neighbors {
                operand,
                direction: drection,
            } => Box::new(Self::evaluate_neighbors(
                medrecord,
                node_indices,
                operand,
                drection.clone(),
            )?),
            Self::EitherOr { either, or } => {
                let (node_indices_1, rest) = node_indices.tee();
                let (node_indices_2, node_indices_3) = rest.tee();

                let either_set: HashSet<&NodeIndex> = either
                    .evaluate_forward(medrecord, Box::new(node_indices_1))?
                    .collect();
                let or_set: HashSet<&NodeIndex> = or
                    .evaluate_forward(medrecord, Box::new(node_indices_2))?
                    .collect();

                Box::new(node_indices_3.filter(move |node_index| {
                    either_set.contains(node_index) || or_set.contains(node_index)
                }))
            }
            Self::Exclude { operand } => {
                let (node_indices_1, node_indices_2) = node_indices.tee();

                let result: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(node_indices_1))?
                    .collect();

                Box::new(node_indices_2.filter(move |node_index| !result.contains(node_index)))
            }
            Self::GroupBy { operand } => {
                Box::new(Self::evaluate_group_by(medrecord, node_indices, operand)?)
            }
        })
    }

    #[inline]
    pub(crate) fn get_values<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute: MedRecordAttribute,
    ) -> impl Iterator<Item = (&'a NodeIndex, MedRecordValue)> {
        node_indices.flat_map(move |node_index| {
            Some((
                node_index,
                medrecord
                    .node_attributes(node_index)
                    .expect("Edge must exist")
                    .get(&attribute)?
                    .clone(),
            ))
        })
    }

    #[inline]
    fn evaluate_values<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: &Wrapper<MultipleValuesWithIndexOperand<NodeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let MultipleValuesWithIndexContext::Operand((_, ref attribute)) =
            operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let values = Self::get_values(medrecord, node_indices, attribute.clone());

        Ok(operand
            .evaluate_forward(medrecord, Box::new(values))?
            .map(|value| value.0))
    }

    #[inline]
    pub(crate) fn get_attributes<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
    ) -> impl Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)> {
        node_indices.map(move |node_index| {
            let attributes = medrecord
                .node_attributes(node_index)
                .expect("Edge must exist")
                .keys()
                .cloned();

            (node_index, attributes.collect())
        })
    }

    #[inline]
    fn evaluate_attributes<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: &Wrapper<AttributesTreeOperand<NodeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let attributes = Self::get_attributes(medrecord, node_indices);

        Ok(operand
            .evaluate_forward(medrecord, Box::new(attributes))?
            .map(|value| value.0))
    }

    #[inline]
    fn evaluate_indices<'a>(
        medrecord: &MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operand: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(node_indices_1.cloned()))?
            .collect();

        Ok(node_indices_2
            .into_iter()
            .filter(move |index| result.contains(*index)))
    }

    #[inline]
    fn evaluate_in_group<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        group: CardinalityWrapper<Group>,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices.filter(move |node_index| {
            let groups_of_node = medrecord
                .groups_of_node(node_index)
                .expect("Node must exist");

            let groups_of_node: MrHashSet<_> = groups_of_node.collect();

            match &group {
                CardinalityWrapper::Single(group) => groups_of_node.contains(&group),
                CardinalityWrapper::Multiple(groups) => {
                    groups.iter().all(|group| groups_of_node.contains(&group))
                }
            }
        })
    }

    #[inline]
    fn evaluate_has_attribute<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        attribute: CardinalityWrapper<MedRecordAttribute>,
    ) -> impl Iterator<Item = &'a NodeIndex> {
        node_indices.filter(move |node_index| {
            let attributes_of_node = medrecord
                .node_attributes(node_index)
                .expect("Node must exist")
                .keys();

            let attributes_of_node: MrHashSet<_> = attributes_of_node.collect();

            match &attribute {
                CardinalityWrapper::Single(attribute) => attributes_of_node.contains(&attribute),
                CardinalityWrapper::Multiple(attributes) => attributes
                    .iter()
                    .all(|attribute| attributes_of_node.contains(&attribute)),
            }
        })
    }

    #[inline]
    fn evaluate_edges<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operand: &Wrapper<EdgeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

        let edge_indices: BoxedIterator<_> = match direction {
            EdgeDirection::Incoming => Box::new(node_indices_1.flat_map(|node_index| {
                medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist.")
            })),
            EdgeDirection::Outgoing => Box::new(node_indices_1.flat_map(|node_index| {
                medrecord
                    .outgoing_edges(node_index)
                    .expect("Node must exist.")
            })),
            EdgeDirection::Both => Box::new(node_indices_1.flat_map(|node_index| {
                medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist")
                    .chain(
                        medrecord
                            .outgoing_edges(node_index)
                            .expect("Node must exist"),
                    )
            })),
        };

        let edge_indices: RoaringBitmap =
            operand.evaluate_forward(medrecord, edge_indices)?.collect();

        Ok(node_indices_2.filter(move |node_index| {
            let connected_indices: RoaringBitmap = match direction {
                EdgeDirection::Incoming => medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist")
                    .collect(),
                EdgeDirection::Outgoing => medrecord
                    .outgoing_edges(node_index)
                    .expect("Node must exist")
                    .collect(),
                EdgeDirection::Both => medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist")
                    .chain(
                        medrecord
                            .outgoing_edges(node_index)
                            .expect("Node must exist"),
                    )
                    .collect(),
            };

            !connected_indices.is_disjoint(&edge_indices)
        }))
    }

    #[inline]
    fn evaluate_neighbors<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: &Wrapper<NodeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

        let neighbor_indices: BoxedIterator<_> = match direction {
            EdgeDirection::Incoming => Box::new(node_indices_1.flat_map(move |node_index| {
                medrecord
                    .neighbors_incoming(node_index)
                    .expect("Node must exist")
            })),
            EdgeDirection::Outgoing => Box::new(node_indices_1.flat_map(move |node_index| {
                medrecord
                    .neighbors_outgoing(node_index)
                    .expect("Node must exist")
            })),
            EdgeDirection::Both => Box::new(node_indices_1.flat_map(move |node_index| {
                medrecord
                    .neighbors_undirected(node_index)
                    .expect("Node must exist")
            })),
        };

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, neighbor_indices)?
            .collect();

        Ok(node_indices_2.filter(move |node_index| {
            let mut neighbors: BoxedIterator<_> = match direction {
                EdgeDirection::Incoming => Box::new(
                    medrecord
                        .neighbors_incoming(node_index)
                        .expect("Node must exist"),
                ),
                EdgeDirection::Outgoing => Box::new(
                    medrecord
                        .neighbors_outgoing(node_index)
                        .expect("Node must exist"),
                ),
                EdgeDirection::Both => Box::new(
                    medrecord
                        .neighbors_undirected(node_index)
                        .expect("Node must exist"),
                ),
            };

            neighbors.any(|neighbor| result.contains(&neighbor))
        }))
    }

    fn evaluate_group_by<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
        operand: &Wrapper<GroupOperand<NodeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        Ok(NodeOperand::merge(
            operand.evaluate_forward(medrecord, Box::new(node_indices))?,
        ))
    }
}

impl NodeOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        Ok(match self {
            NodeOperation::Values { operand } => {
                Self::evaluate_values_grouped(medrecord, node_indices, operand)?
            }
            NodeOperation::Attributes { operand } => {
                Self::evaluate_attributes_grouped(medrecord, node_indices, operand)?
            }
            NodeOperation::Indices { operand } => {
                Self::evaluate_indices_grouped(medrecord, node_indices, operand)?
            }
            NodeOperation::InGroup { group } => {
                let group = group.clone();

                Box::new(node_indices.map(move |(key, node_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_in_group(
                            medrecord,
                            node_indices,
                            group.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            NodeOperation::HasAttribute { attribute } => {
                let attribute = attribute.clone();

                Box::new(node_indices.map(move |(key, node_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_has_attribute(
                            medrecord,
                            node_indices,
                            attribute.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            NodeOperation::Edges { operand, direction } => {
                Self::evaluate_edges_grouped(medrecord, node_indices, operand, direction.clone())?
            }
            NodeOperation::Neighbors { operand, direction } => Self::evaluate_neighbors_grouped(
                medrecord,
                node_indices,
                operand,
                direction.clone(),
            )?,
            NodeOperation::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, node_indices, either, or)?
            }
            NodeOperation::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, node_indices, operand)?
            }
            NodeOperation::GroupBy { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    fn evaluate_values_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<MultipleValuesWithIndexOperand<NodeOperand>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let MultipleValuesWithIndexContext::Operand((_, ref attribute)) =
            operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let values: Vec<_> = node_indices
            .map(|(key, node_indices)| {
                (
                        key,
                        Box::new(Self::get_values(medrecord, node_indices, attribute.clone()))
                            as <MultipleValuesWithIndexOperand<NodeOperand> as EvaluateForward<
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
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<AttributesTreeOperand<NodeOperand>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let attributes = node_indices.map(|(key, node_indices)| {
            (
                key,
                Box::new(Self::get_attributes(medrecord, node_indices))
                    as <AttributesTreeOperand<NodeOperand> as EvaluateForward<'a>>::InputValue,
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
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let node_indices_1 = node_indices_1
            .map(|(key, node_indices)| (key, Box::new(node_indices.cloned()) as BoxedIterator<_>));

        let mut node_indices_1: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(node_indices_1))?
            .collect();

        Ok(Box::new(node_indices_2.map(move |(key, node_indices)| {
            let node_indices_position = &node_indices_1
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let node_indices_1: MrHashSet<_> =
                node_indices_1.remove(*node_indices_position).1.collect();

            let filtered_indices: Vec<_> = node_indices
                .filter(|node_index| node_indices_1.contains(*node_index))
                .collect();

            (
                key,
                Box::new(filtered_indices.into_iter()) as BoxedIterator<_>,
            )
        })))
    }

    #[inline]
    fn evaluate_edges_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<EdgeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let direction_clone = direction.clone();

        let edge_indices = node_indices_1.map(move |(key, node_indices)| {
            let edge_indices: BoxedIterator<_> = match direction_clone {
                EdgeDirection::Incoming => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .incoming_edges(node_index)
                        .expect("Node must exist")
                })),
                EdgeDirection::Outgoing => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .outgoing_edges(node_index)
                        .expect("Node must exist")
                })),
                EdgeDirection::Both => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .incoming_edges(node_index)
                        .expect("Node must exist")
                        .chain(
                            medrecord
                                .outgoing_edges(node_index)
                                .expect("Node must exist"),
                        )
                })),
            };

            (key, edge_indices)
        });

        let mut edge_indices: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(edge_indices))?
            .collect();

        Ok(Box::new(node_indices_2.map(move |(key, node_indices)| {
            let edge_indices_position = &edge_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let edge_indices: RoaringBitmap =
                edge_indices.remove(*edge_indices_position).1.collect();

            let filtered_indices: Vec<_> = node_indices
                .filter(|node_index| {
                    let connected_indices: RoaringBitmap = match direction.clone() {
                        EdgeDirection::Incoming => medrecord
                            .incoming_edges(node_index)
                            .expect("Node must exist")
                            .collect(),
                        EdgeDirection::Outgoing => medrecord
                            .outgoing_edges(node_index)
                            .expect("Node must exist")
                            .collect(),
                        EdgeDirection::Both => medrecord
                            .incoming_edges(node_index)
                            .expect("Node must exist")
                            .chain(
                                medrecord
                                    .outgoing_edges(node_index)
                                    .expect("Node must exist"),
                            )
                            .collect(),
                    };

                    !connected_indices.is_disjoint(&edge_indices)
                })
                .collect();

            (
                key,
                Box::new(filtered_indices.into_iter()) as BoxedIterator<_>,
            )
        })))
    }

    #[inline]
    fn evaluate_neighbors_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<NodeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let direction_clone = direction.clone();

        let neighbor_indices = node_indices_1.map(move |(key, node_indices)| {
            let neighbor_indices: BoxedIterator<_> = match direction_clone {
                EdgeDirection::Incoming => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .neighbors_incoming(node_index)
                        .expect("Node must exist")
                })),
                EdgeDirection::Outgoing => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .neighbors_outgoing(node_index)
                        .expect("Node must exist")
                })),
                EdgeDirection::Both => Box::new(node_indices.flat_map(move |node_index| {
                    medrecord
                        .neighbors_undirected(node_index)
                        .expect("Node must exist")
                })),
            };

            (key, neighbor_indices)
        });

        let mut neighbor_indices: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(neighbor_indices))?
            .collect();

        Ok(Box::new(node_indices_2.map(move |(key, node_indices)| {
            let neighbor_indices_position = &neighbor_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let neighbor_indices: MrHashSet<_> = neighbor_indices
                .remove(*neighbor_indices_position)
                .1
                .collect();

            let filtered_indices: Vec<_> = node_indices
                .filter(|node_index| {
                    let mut neighbors: Box<dyn Iterator<Item = &MedRecordAttribute>> =
                        match direction {
                            EdgeDirection::Incoming => Box::new(
                                medrecord
                                    .neighbors_incoming(node_index)
                                    .expect("Node must exist"),
                            ),
                            EdgeDirection::Outgoing => Box::new(
                                medrecord
                                    .neighbors_outgoing(node_index)
                                    .expect("Node must exist"),
                            ),
                            EdgeDirection::Both => Box::new(
                                medrecord
                                    .neighbors_undirected(node_index)
                                    .expect("Node must exist"),
                            ),
                        };

                    neighbors.any(|neighbor| neighbor_indices.contains(&neighbor))
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
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        either: &Wrapper<NodeOperand>,
        or: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let either_indices = either.evaluate_forward_grouped(medrecord, node_indices_1)?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, node_indices_2)?
            .collect();

        let node_indices = either_indices.map(move |(key, either_indices)| {
            let indices_position = or_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_indices = or_indices.remove(indices_position).1;

            let node_indices: BoxedIterator<_> = Box::new(
                either_indices
                    .chain(or_indices)
                    .unique_by(|node_index| (*node_index).clone()),
            );

            (key, node_indices)
        });

        Ok(Box::new(node_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, node_indices_1)?
            .collect();

        let node_indices = node_indices_2.map(move |(key, values)| {
            let indices_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_indices: MrHashSet<_> = result.remove(indices_position).1.collect();

            let node_indices: BoxedIterator<_> =
                Box::new(values.filter(move |node_index| !excluded_indices.contains(node_index)));

            (key, node_indices)
        });

        Ok(Box::new(node_indices))
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndicesOperation {
    NodeIndexOperation {
        operand: Wrapper<NodeIndexOperand>,
    },
    NodeIndexComparisonOperation {
        operand: NodeIndexComparisonOperand,
        kind: SingleComparisonKind,
    },
    NodeIndicesComparisonOperation {
        operand: NodeIndicesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: NodeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<NodeIndicesOperand>,
        or: Wrapper<NodeIndicesOperand>,
    },
    Exclude {
        operand: Wrapper<NodeIndicesOperand>,
    },

    Merge {
        operand: Wrapper<NodeIndicesOperand>,
    },
}

impl DeepClone for NodeIndicesOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeIndexOperation { operand } => Self::NodeIndexOperation {
                operand: operand.deep_clone(),
            },
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::NodeIndexComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::NodeIndicesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::UnaryArithmeticOperation { kind } => {
                Self::UnaryArithmeticOperation { kind: kind.clone() }
            }
            Self::Slice(range) => Self::Slice(range.clone()),
            Self::IsString => Self::IsString,
            Self::IsInt => Self::IsInt,
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

impl NodeIndicesOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        Ok(match self {
            Self::NodeIndexOperation { operand } => {
                Self::evaluate_node_index_operation(medrecord, indices, operand)?
            }
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_node_index_comparison_operation(medrecord, indices, operand, kind)?
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_node_indices_comparison_operation(medrecord, indices, operand, kind)?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, indices, operand, kind)?,
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(indices, kind.clone()),
            ),
            Self::Slice(range) => Box::new(Self::evaluate_slice(indices, range.clone())),
            Self::IsString => Box::new(Self::evaluate_is_string(indices)),
            Self::IsInt => Box::new(Self::evaluate_is_int(indices)),
            Self::IsMax => Self::evaluate_is_max(indices)?,
            Self::IsMin => Self::evaluate_is_min(indices)?,
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, indices, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, indices, operand)?,
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    pub(crate) fn get_max(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let max_index = indices.next();

        let Some(max_index) = max_index else {
            return Ok(None);
        };

        let max_index = indices.try_fold(max_index, |max_index, index| {
            match index
            .partial_cmp(&max_index) {
                Some(Ordering::Greater) => Ok(index),
                None => {
                    let first_dtype = DataType::from(index);
                    let second_dtype = DataType::from(max_index);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare indices of data types {first_dtype} and {second_dtype}. Consider narrowing down the indices using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(max_index),
            }
        })?;

        Ok(Some(max_index))
    }

    #[inline]
    pub(crate) fn get_min(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let min_index = indices.next();

        let Some(min_index) = min_index else {
            return Ok(None);
        };

        let min_index = indices.try_fold(min_index, |min_index, index| {
            match index.partial_cmp(&min_index) {
                Some(Ordering::Less) => Ok(index),
                None => {
                    let first_dtype = DataType::from(index);
                    let second_dtype = DataType::from(min_index);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare indices of data types {first_dtype} and {second_dtype}. Consider narrowing down the indices using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(min_index),
            }
        })?;

        Ok(Some(min_index))
    }
    #[inline]
    pub(crate) fn get_count(indices: impl Iterator<Item = NodeIndex>) -> NodeIndex {
        MedRecordAttribute::Int(indices.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let first_index = indices.next();

        let Some(first_index) = first_index else {
            return Ok(None);
        };

        let sum = indices.try_fold(first_index, |sum, index| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&index);

            sum.add(index).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add indices of data types {first_dtype} and {second_dtype}. Consider narrowing down the indices using .is_string() or .is_int()"
                ))
            })
        })?;

        Ok(Some(sum))
    }

    #[inline]
    pub(crate) fn get_random(indices: impl Iterator<Item = NodeIndex>) -> Option<NodeIndex> {
        indices.choose(&mut rng())
    }

    #[inline]
    fn evaluate_node_index_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
        operand: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let kind = &operand.0.read_or_panic().kind;

        let index = match kind {
            SingleKind::Max => NodeIndicesOperation::get_max(indices_1)?,
            SingleKind::Min => NodeIndicesOperation::get_min(indices_1)?,
            SingleKind::Count => Some(NodeIndicesOperation::get_count(indices_1)),
            SingleKind::Sum => NodeIndicesOperation::get_sum(indices_1)?,
            SingleKind::Random => NodeIndicesOperation::get_random(indices_1),
        };

        Ok(match operand.evaluate_forward(medrecord, index)? {
            Some(_) => Box::new(indices_2.into_iter()),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_node_index_comparison_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
        comparison_operand: &NodeIndexComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
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
    fn evaluate_node_indices_comparison_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
        comparison_operand: &NodeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
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
        indices: impl Iterator<Item = NodeIndex>,
        operand: &NodeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex>> {
        let arithmetic_index =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        let indices = indices
            .map(move |index| {
                match kind {
                    BinaryArithmeticKind::Add => index.add(arithmetic_index.clone()),
                    BinaryArithmeticKind::Sub => index.sub(arithmetic_index.clone()),
                    BinaryArithmeticKind::Mul => {
                        index.clone().mul(arithmetic_index.clone())
                    }
                    BinaryArithmeticKind::Pow => {
                        index.clone().pow(arithmetic_index.clone())
                    }
                    BinaryArithmeticKind::Mod => {
                        index.clone().r#mod(arithmetic_index.clone())
                    }
                }
                .map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Failed arithmetic operation {kind}. Consider narrowing down the indices using .is_string() or .is_int()",
                    ))
                })
            });

        Ok(indices.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation(
        indices: impl Iterator<Item = NodeIndex>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = NodeIndex> {
        indices.map(move |index| match kind {
            UnaryArithmeticKind::Abs => index.abs(),
            UnaryArithmeticKind::Trim => index.trim(),
            UnaryArithmeticKind::TrimStart => index.trim_start(),
            UnaryArithmeticKind::TrimEnd => index.trim_end(),
            UnaryArithmeticKind::Lowercase => index.lowercase(),
            UnaryArithmeticKind::Uppercase => index.uppercase(),
        })
    }

    #[inline]
    fn evaluate_slice(
        indices: impl Iterator<Item = NodeIndex>,
        range: Range<usize>,
    ) -> impl Iterator<Item = NodeIndex> {
        indices.map(move |index| index.slice(range.clone()))
    }

    #[inline]
    fn evaluate_is_string(
        indices: impl Iterator<Item = NodeIndex>,
    ) -> impl Iterator<Item = NodeIndex> {
        indices.filter(|index| matches!(index, MedRecordAttribute::String(_)))
    }

    #[inline]
    fn evaluate_is_int(
        indices: impl Iterator<Item = NodeIndex>,
    ) -> impl Iterator<Item = NodeIndex> {
        indices.filter(|index| matches!(index, MedRecordAttribute::Int(_)))
    }

    #[inline]
    fn evaluate_is_max<'a>(
        indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let max_index = Self::get_max(indices_1)?;

        let Some(max_index) = max_index else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(indices_2.filter(move |index| *index == max_index)))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let min_index = Self::get_min(indices_1)?;

        let Some(min_index) = min_index else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(indices_2.filter(move |index| *index == min_index)))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
        either: &Wrapper<NodeIndicesOperand>,
        or: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let either_indices = either.evaluate_forward(medrecord, Box::new(indices_1))?;
        let or_indices = or.evaluate_forward(medrecord, Box::new(indices_2))?;

        Ok(Box::new(either_indices.chain(or_indices).unique()))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
        operand: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let (indices_1, indices_2) = Itertools::tee(indices);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(indices_1))?
            .collect();

        Ok(Box::new(
            indices_2
                .filter(move |index| !result.contains(index))
                .unique(),
        ))
    }
}

impl NodeIndicesOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>> {
        Ok(match self {
            NodeIndicesOperation::NodeIndexOperation { operand } => {
                Self::evaluate_node_index_operation_grouped(medrecord, node_indices, operand)?
            }
            NodeIndicesOperation::NodeIndexComparisonOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_node_index_comparison_operation(
                                medrecord,
                                node_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndicesOperation::NodeIndicesComparisonOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_node_indices_comparison_operation(
                                medrecord,
                                node_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndicesOperation::BinaryArithmeticOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord,
                                node_indices,
                                operand,
                                kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndicesOperation::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(node_indices.map(move |(key, node_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_unary_arithmetic_operation(
                            node_indices,
                            kind.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            NodeIndicesOperation::Slice(range) => {
                let range = range.clone();

                Box::new(node_indices.map(move |(key, node_indices)| {
                    (
                        key,
                        Box::new(Self::evaluate_slice(node_indices, range.clone()))
                            as BoxedIterator<_>,
                    )
                }))
            }
            NodeIndicesOperation::IsString => Box::new(node_indices.map(|(key, node_indices)| {
                (
                    key,
                    Box::new(Self::evaluate_is_string(node_indices)) as BoxedIterator<_>,
                )
            })),
            NodeIndicesOperation::IsInt => Box::new(node_indices.map(|(key, node_indices)| {
                (
                    key,
                    Box::new(Self::evaluate_is_int(node_indices)) as BoxedIterator<_>,
                )
            })),
            NodeIndicesOperation::IsMax => Box::new(
                node_indices
                    .map(move |(key, node_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_max(node_indices)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndicesOperation::IsMin => Box::new(
                node_indices
                    .map(move |(key, node_indices)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_min(node_indices)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndicesOperation::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, node_indices, either, or)?
            }
            NodeIndicesOperation::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, node_indices, operand)?
            }
            NodeIndicesOperation::Merge { operand } => {
                let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

                let node_indices_1 = node_indices_1.flat_map(|(_, value)| value);

                let node_indinces_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(node_indices_1))?
                    .collect();

                Box::new(node_indices_2.map(move |(key, node_indices)| {
                    let node_indices: Vec<_> = node_indices
                        .filter(|node_index| node_indinces_1.contains(node_index))
                        .collect();

                    let node_indices: BoxedIterator<_> = Box::new(node_indices.into_iter());

                    (key, node_indices)
                }))
            }
        })
    }

    #[inline]
    fn evaluate_node_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>,
        operand: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);
        let mut node_indices_2: Vec<_> = node_indices_2.collect();

        let kind = &operand.0.read_or_panic().kind;

        let node_indices_1: Vec<_> = node_indices_1
            .map(move |(key, node_indices)| {
                Ok((
                    key,
                    match kind {
                        SingleKind::Max => NodeIndicesOperation::get_max(node_indices)?,
                        SingleKind::Min => NodeIndicesOperation::get_min(node_indices)?,
                        SingleKind::Count => Some(NodeIndicesOperation::get_count(node_indices)),
                        SingleKind::Sum => NodeIndicesOperation::get_sum(node_indices)?,
                        SingleKind::Random => NodeIndicesOperation::get_random(node_indices),
                    },
                ))
            })
            .collect::<MedRecordResult<_>>()?;

        let node_indices_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(node_indices_1.into_iter()))?;

        Ok(Box::new(node_indices_1.map(
            move |(key, value)| match value {
                Some(_) => {
                    let node_indices_position = node_indices_2
                        .iter()
                        .position(|(k, _)| k == &key)
                        .expect("Entry must exist");

                    node_indices_2.remove(node_indices_position)
                }
                None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
            },
        )))
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>,
        either: &Wrapper<NodeIndicesOperand>,
        or: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let either_indices = either.evaluate_forward_grouped(medrecord, node_indices_1)?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, node_indices_2)?
            .collect();

        let node_indices = either_indices.map(move |(key, either_indices)| {
            let indices_position = or_indices
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_indices = or_indices.remove(indices_position).1;

            let node_indices: BoxedIterator<_> = Box::new(
                either_indices
                    .chain(or_indices)
                    .unique_by(|node_index| node_index.clone()),
            );

            (key, node_indices)
        });

        Ok(Box::new(node_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>,
        operand: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, NodeIndex>>> {
        let (node_indices_1, node_indices_2) = tee_grouped_iterator(node_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, node_indices_1)?
            .collect();

        let node_indices = node_indices_2.map(move |(key, values)| {
            let indices_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_indices: MrHashSet<_> = result.remove(indices_position).1.collect();

            let node_indices: BoxedIterator<_> =
                Box::new(values.filter(move |node_index| !excluded_indices.contains(node_index)));

            (key, node_indices)
        });

        Ok(Box::new(node_indices))
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndexOperation {
    NodeIndexComparisonOperation {
        operand: NodeIndexComparisonOperand,
        kind: SingleComparisonKind,
    },
    NodeIndicesComparisonOperation {
        operand: NodeIndicesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: NodeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    EitherOr {
        either: Wrapper<NodeIndexOperand>,
        or: Wrapper<NodeIndexOperand>,
    },
    Exclude {
        operand: Wrapper<NodeIndexOperand>,
    },

    Merge {
        operand: Wrapper<NodeIndicesOperand>,
    },
}

impl DeepClone for NodeIndexOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::NodeIndexComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::NodeIndicesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::UnaryArithmeticOperation { kind } => {
                Self::UnaryArithmeticOperation { kind: kind.clone() }
            }
            Self::Slice(range) => Self::Slice(range.clone()),
            Self::IsString => Self::IsString,
            Self::IsInt => Self::IsInt,
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

impl NodeIndexOperation {
    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        node_index: Option<NodeIndex>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let Some(node_index) = node_index else {
            return Ok(None);
        };

        Ok(match self {
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_node_index_comparison_operation(
                    medrecord, node_index, operand, kind,
                )?
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_node_indices_comparison_operation(
                    medrecord, node_index, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, node_index, operand, kind)?
            }
            Self::UnaryArithmeticOperation { kind } => {
                Some(Self::evaluate_unary_arithmetic_operation(node_index, kind))
            }
            Self::Slice(range) => Some(Self::evaluate_slice(node_index, range)),
            Self::IsString => Self::evaluate_is_string(node_index),
            Self::IsInt => Self::evaluate_is_int(node_index),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, node_index, either, or)?
            }
            Self::Exclude { operand } => {
                let result = operand
                    .evaluate_forward(medrecord, Some(node_index.clone()))?
                    .is_some();

                if result {
                    None
                } else {
                    Some(node_index)
                }
            }
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    fn evaluate_node_index_comparison_operation(
        medrecord: &MedRecord,
        index: NodeIndex,
        comparison_operand: &NodeIndexComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<NodeIndex>> {
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
    fn evaluate_node_indices_comparison_operation(
        medrecord: &MedRecord,
        index: NodeIndex,
        comparison_operand: &NodeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<NodeIndex>> {
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
        index: NodeIndex,
        operand: &NodeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let arithmetic_index =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No index to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => index.add(arithmetic_index)?,
            BinaryArithmeticKind::Sub => index.sub(arithmetic_index)?,
            BinaryArithmeticKind::Mul => index.mul(arithmetic_index)?,
            BinaryArithmeticKind::Pow => index.pow(arithmetic_index)?,
            BinaryArithmeticKind::Mod => index.r#mod(arithmetic_index)?,
        }))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation(
        node_index: NodeIndex,
        kind: &UnaryArithmeticKind,
    ) -> MedRecordAttribute {
        match kind {
            UnaryArithmeticKind::Abs => node_index.abs(),
            UnaryArithmeticKind::Trim => node_index.trim(),
            UnaryArithmeticKind::TrimStart => node_index.trim_start(),
            UnaryArithmeticKind::TrimEnd => node_index.trim_end(),
            UnaryArithmeticKind::Lowercase => node_index.lowercase(),
            UnaryArithmeticKind::Uppercase => node_index.uppercase(),
        }
    }

    #[inline]
    fn evaluate_slice(node_index: NodeIndex, range: &Range<usize>) -> NodeIndex {
        node_index.slice(range.clone())
    }

    #[inline]
    fn evaluate_is_string(node_index: NodeIndex) -> Option<NodeIndex> {
        match node_index {
            NodeIndex::String(_) => Some(node_index),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_int(node_index: NodeIndex) -> Option<NodeIndex> {
        match node_index {
            NodeIndex::Int(_) => Some(node_index),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_either_or(
        medrecord: &MedRecord,
        index: NodeIndex,
        either: &Wrapper<NodeIndexOperand>,
        or: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let either_result = either.evaluate_forward(medrecord, Some(index.clone()))?;
        let or_result = or.evaluate_forward(medrecord, Some(index))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl NodeIndexOperation {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, Option<NodeIndex>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<NodeIndex>>> {
        Ok(match self {
            NodeIndexOperation::NodeIndexComparisonOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_index)| {
                        let Some(node_index) = node_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_node_index_comparison_operation(
                                medrecord, node_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndexOperation::NodeIndicesComparisonOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_index)| {
                        let Some(node_index) = node_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_node_indices_comparison_operation(
                                medrecord, node_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndexOperation::BinaryArithmeticOperation { operand, kind } => Box::new(
                node_indices
                    .map(move |(key, node_index)| {
                        let Some(node_index) = node_index else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, node_index, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            NodeIndexOperation::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(node_indices.map(move |(key, node_index)| {
                    let Some(node_index) = node_index else {
                        return (key, None);
                    };

                    (
                        key,
                        Some(Self::evaluate_unary_arithmetic_operation(node_index, &kind)),
                    )
                }))
            }
            NodeIndexOperation::Slice(range) => {
                let range = range.clone();

                Box::new(node_indices.map(move |(key, node_index)| {
                    let Some(node_index) = node_index else {
                        return (key, None);
                    };

                    (key, Some(Self::evaluate_slice(node_index, &range)))
                }))
            }
            NodeIndexOperation::IsString => Box::new(node_indices.map(move |(key, node_index)| {
                let Some(node_index) = node_index else {
                    return (key, None);
                };

                (key, Self::evaluate_is_string(node_index))
            })),
            NodeIndexOperation::IsInt => Box::new(node_indices.map(move |(key, node_index)| {
                let Some(node_index) = node_index else {
                    return (key, None);
                };

                (key, Self::evaluate_is_int(node_index))
            })),
            NodeIndexOperation::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, node_indices, either, or)?
            }
            NodeIndexOperation::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, node_indices, operand)?
            }
            NodeIndexOperation::Merge { operand } => {
                let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

                let node_indices_1 = node_indices_1.filter_map(|(_, node_index)| node_index);

                let node_indices_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(node_indices_1))?
                    .collect();

                Box::new(node_indices_2.map(move |(key, node_index)| {
                    let node_index =
                        node_index.filter(|node_index| node_indices_1.contains(node_index));

                    (key, node_index)
                }))
            }
        })
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, Option<NodeIndex>>,
        either: &Wrapper<NodeIndexOperand>,
        or: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<NodeIndex>>> {
        let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

        let either_indices =
            either.evaluate_forward_grouped(medrecord, Box::new(node_indices_1))?;
        let mut or_indices: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(node_indices_2))?
            .collect();

        let node_indices = either_indices.map(move |(key, either_indices)| {
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

        Ok(Box::new(node_indices))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, Option<NodeIndex>>,
        operand: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<NodeIndex>>> {
        let (node_indices_1, node_indices_2) = Itertools::tee(node_indices);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(node_indices_1))?
            .collect();

        let node_indices = node_indices_2.map(move |(key, node_index)| {
            let index_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_index = result.remove(index_position).1;

            let node_index = match excluded_index {
                Some(_) => None,
                None => node_index,
            };

            (key, node_index)
        });

        Ok(Box::new(node_indices))
    }
}

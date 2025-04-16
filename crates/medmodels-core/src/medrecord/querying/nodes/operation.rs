use super::{
    operand::{
        NodeIndexComparisonOperand, NodeIndexOperand, NodeIndicesComparisonOperand,
        NodeIndicesOperand,
    },
    BinaryArithmeticKind, MultipleComparisonKind, NodeOperand, SingleComparisonKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Contains, EndsWith, Lowercase, Mod, Pow, Slice, StartsWith, Trim, TrimEnd,
            TrimStart, Uppercase,
        },
        querying::{
            attributes::AttributesTreeOperand,
            edges::EdgeOperand,
            traits::DeepClone,
            values::MultipleValuesOperand,
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator,
        },
        DataType, Group, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex,
    },
};
use itertools::Itertools;
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
        operand: Wrapper<MultipleValuesOperand<NodeOperand>>,
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
        }
    }
}

impl NodeOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        Ok(match self {
            Self::Values { operand } => {
                Box::new(Self::evaluate_values(medrecord, operand.clone())?)
            }
            Self::Attributes { operand } => {
                Box::new(Self::evaluate_attributes(medrecord, operand.clone())?)
            }
            Self::Indices { operand } => {
                Box::new(Self::evaluate_indices(medrecord, operand.clone())?)
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
                operand.clone(),
                direction.clone(),
            )?),
            Self::Neighbors {
                operand,
                direction: drection,
            } => Box::new(Self::evaluate_neighbors(
                medrecord,
                node_indices,
                operand.clone(),
                drection.clone(),
            )?),
            Self::EitherOr { either, or } => {
                // TODO: This is a temporary solution. It should be optimized.
                let either_result = either.evaluate(medrecord)?.collect::<HashSet<_>>();
                let or_result = or.evaluate(medrecord)?.collect::<HashSet<_>>();

                Box::new(node_indices.filter(move |node_index| {
                    either_result.contains(node_index) || or_result.contains(node_index)
                }))
            }
            Self::Exclude { operand } => {
                let result = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

                Box::new(node_indices.filter(move |node_index| !result.contains(node_index)))
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
        operand: Wrapper<MultipleValuesOperand<NodeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex>> {
        Ok(operand.evaluate(medrecord)?.map(|value| value.0))
    }

    #[inline]
    pub(crate) fn get_attributes<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> impl Iterator<Item = (NodeIndex, Vec<MedRecordAttribute>)> + 'a {
        node_indices.map(move |node_index| {
            let attributes = medrecord
                .node_attributes(&node_index)
                .expect("Edge must exist")
                .keys()
                .cloned();

            (node_index, attributes.collect())
        })
    }

    #[inline]
    fn evaluate_attributes<'a>(
        medrecord: &'a MedRecord,
        operand: Wrapper<AttributesTreeOperand<NodeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        Ok(operand.evaluate(medrecord)?.map(|value| value.0))
    }

    #[inline]
    fn evaluate_indices<'a>(
        medrecord: &'a MedRecord,
        operand: Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        operand.evaluate(medrecord)
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

            let groups_of_node = groups_of_node.collect::<Vec<_>>();

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

            let attributes_of_node = attributes_of_node.collect::<Vec<_>>();

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
        operand: Wrapper<EdgeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let edge_indices = operand.evaluate(medrecord)?.collect::<RoaringBitmap>();

        Ok(node_indices.filter(move |node_index| {
            let connected_indices = match direction {
                EdgeDirection::Incoming => medrecord
                    .outgoing_edges(node_index)
                    .expect("Node must exist")
                    .collect::<RoaringBitmap>(),
                EdgeDirection::Outgoing => medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist")
                    .collect::<RoaringBitmap>(),
                EdgeDirection::Both => medrecord
                    .incoming_edges(node_index)
                    .expect("Node must exist")
                    .chain(
                        medrecord
                            .outgoing_edges(node_index)
                            .expect("Node must exist"),
                    )
                    .collect::<RoaringBitmap>(),
            };

            !connected_indices.is_disjoint(&edge_indices)
        }))
    }

    #[inline]
    fn evaluate_neighbors<'a>(
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex>,
        operand: Wrapper<NodeOperand>,
        direction: EdgeDirection,
    ) -> MedRecordResult<impl Iterator<Item = &'a NodeIndex>> {
        let result = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

        Ok(node_indices.filter(move |node_index| {
            let mut neighbors: Box<dyn Iterator<Item = &MedRecordAttribute>> = match direction {
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
    BinaryArithmeticOpration {
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
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
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
        }
    }
}

impl NodeIndicesOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        match self {
            Self::NodeIndexOperation { operand } => {
                Self::evaluate_node_index_operation(medrecord, indices, operand)
            }
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_node_index_comparison_operation(medrecord, indices, operand, kind)
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_node_indices_comparison_operation(medrecord, indices, operand, kind)
            }
            Self::BinaryArithmeticOpration { operand, kind } => {
                Ok(Box::new(Self::evaluate_binary_arithmetic_operation(
                    medrecord,
                    indices,
                    operand,
                    kind.clone(),
                )?))
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Box::new(
                Self::evaluate_unary_arithmetic_operation(indices, kind.clone()),
            )),
            Self::Slice(range) => Ok(Box::new(Self::evaluate_slice(indices, range.clone()))),
            Self::IsString => {
                Ok(Box::new(indices.filter(|index| {
                    matches!(index, MedRecordAttribute::String(_))
                })))
            }
            Self::IsInt => {
                Ok(Box::new(indices.filter(|index| {
                    matches!(index, MedRecordAttribute::Int(_))
                })))
            }
            Self::IsMax => {
                let max_index = Self::get_max(indices)?;

                Ok(Box::new(indices.filter(move |index| index == max_index)))
            }
            Self::IsMin => {
                let min_index = Self::get_min(indices)?;

                Ok(Box::new(indices.filter(move |index| index == min_index)))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, indices, either, or)
            }
            Self::Exclude { operand } => {
                let result = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

                Ok(Box::new(
                    indices.filter(move |index| !result.contains(index)),
                ))
            }
        }
    }

    #[inline]
    pub(crate) fn get_max<'a>(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<NodeIndex> {
        let max_index = indices.next().ok_or(MedRecordError::QueryError(
            "No indices to compare".to_string(),
        ))?;

        indices.try_fold(max_index, |max_index, index| {
            match index
            .partial_cmp(&max_index) {
                Some(Ordering::Greater) => Ok(index),
                None => {
                    let first_dtype = DataType::from(index);
                    let second_dtype = DataType::from(max_index);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare indices of data types {} and {}. Consider narrowing down the indices using .is_string() or .is_int()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(max_index),
            }
        })
    }

    #[inline]
    pub(crate) fn get_min(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<NodeIndex> {
        let min_index = indices.next().ok_or(MedRecordError::QueryError(
            "No indices to compare".to_string(),
        ))?;

        indices.try_fold(min_index, |min_index, index| {
            match index.partial_cmp(&min_index) {
                Some(Ordering::Less) => Ok(index),
                None => {
                    let first_dtype = DataType::from(index);
                    let second_dtype = DataType::from(min_index);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare indices of data types {} and {}. Consider narrowing down the indices using .is_string() or .is_int()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(min_index),
            }
        })
    }
    #[inline]
    pub(crate) fn get_count(indices: impl Iterator<Item = NodeIndex>) -> NodeIndex {
        MedRecordAttribute::Int(indices.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<NodeIndex> {
        let first_index = indices
            .next()
            .ok_or(MedRecordError::QueryError("No indices to sum".to_string()))?;

        indices.try_fold(first_index, |sum, index| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&index);

            sum.add(index).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add indices of data types {} and {}. Consider narrowing down the indices using .is_string() or .is_int()",
                    first_dtype, second_dtype
                ))
            })
        })
    }

    #[inline]
    pub(crate) fn get_first<'a>(
        mut indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<NodeIndex> {
        indices.next().ok_or(MedRecordError::QueryError(
            "No indices to get the first".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_last<'a>(
        indices: impl Iterator<Item = NodeIndex>,
    ) -> MedRecordResult<NodeIndex> {
        indices.last().ok_or(MedRecordError::QueryError(
            "No indices to get the first".to_string(),
        ))
    }

    #[inline]
    fn evaluate_node_index_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = NodeIndex>,
        operand: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        Ok(match operand.evaluate(medrecord)? {
            Some(_) => Box::new(indices.into_iter()),
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
                .evaluate(medrecord)?
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
        match comparison_operand {
            NodeIndicesComparisonOperand::Operand(operand) => {
                let comparison_indices = operand.evaluate(medrecord)?;

                match kind {
                    MultipleComparisonKind::IsIn => Ok(Box::new(
                        indices.filter(move |index| comparison_indices.contains(index)),
                    )),
                    MultipleComparisonKind::IsNotIn => Ok(Box::new(
                        indices.filter(move |index| !comparison_indices.contains(index)),
                    )),
                }
            }
            NodeIndicesComparisonOperand::Indices(comparison_indices) => match kind {
                MultipleComparisonKind::IsIn => Ok(Box::new(
                    indices.filter(move |index| comparison_indices.contains(index)),
                )),
                MultipleComparisonKind::IsNotIn => Ok(Box::new(
                    indices.filter(move |index| !comparison_indices.contains(index)),
                )),
            },
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = NodeIndex>,
        operand: &NodeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex>> {
        let arithmetic_index = operand
            .evaluate(medrecord)?
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
                        "Failed arithmetic operation {}. Consider narrowing down the indices using .is_string() or .is_int()",
                        kind,
                    ))
                })
            });

        // TODO: This is a temporary solution. It should be optimized.
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
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex>,
        either: &Wrapper<NodeIndicesOperand>,
        or: &Wrapper<NodeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, NodeIndex>> {
        let indices = indices.collect::<Vec<_>>();

        let either_indices = either.evaluate(medrecord)?;
        let or_indices = or.evaluate(medrecord)?;

        Ok(Box::new(either_indices.chain(or_indices).unique()))
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
    BinaryArithmeticOpration {
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
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
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
        }
    }
}

impl NodeIndexOperation {
    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        index: NodeIndex,
    ) -> MedRecordResult<Option<NodeIndex>> {
        match self {
            Self::NodeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_node_index_comparison_operation(medrecord, index, operand, kind)
            }
            Self::NodeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_node_indices_comparison_operation(medrecord, index, operand, kind)
            }
            Self::BinaryArithmeticOpration { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, index, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Some(match kind {
                UnaryArithmeticKind::Abs => index.abs(),
                UnaryArithmeticKind::Trim => index.trim(),
                UnaryArithmeticKind::TrimStart => index.trim_start(),
                UnaryArithmeticKind::TrimEnd => index.trim_end(),
                UnaryArithmeticKind::Lowercase => index.lowercase(),
                UnaryArithmeticKind::Uppercase => index.uppercase(),
            })),
            Self::Slice(range) => Ok(Some(index.slice(range.clone()))),
            Self::IsString => Ok(match index {
                MedRecordAttribute::String(_) => Some(index),
                _ => None,
            }),
            Self::IsInt => Ok(match index {
                MedRecordAttribute::Int(_) => Some(index),
                _ => None,
            }),
            Self::EitherOr { either, or } => Self::evaluate_either_or(medrecord, index, either, or),
            Self::Exclude { operand } => {
                let result = operand.evaluate(medrecord)?.is_some();

                Ok(if result { None } else { Some(index) })
            }
        }
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
                .evaluate(medrecord)?
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
        match comparison_operand {
            NodeIndicesComparisonOperand::Operand(operand) => {
                let comparison_indices = operand.evaluate(medrecord)?;

                let comparison_result = match kind {
                    MultipleComparisonKind::IsIn => comparison_indices
                        .into_iter()
                        .any(|comparison_index| index == comparison_index),
                    MultipleComparisonKind::IsNotIn => comparison_indices
                        .into_iter()
                        .all(|comparison_index| index != comparison_index),
                };

                Ok(if comparison_result { Some(index) } else { None })
            }
            NodeIndicesComparisonOperand::Indices(comparison_indices) => {
                let comparison_result = match kind {
                    MultipleComparisonKind::IsIn => comparison_indices
                        .into_iter()
                        .any(|comparison_index| index == comparison_index),
                    MultipleComparisonKind::IsNotIn => comparison_indices
                        .into_iter()
                        .all(|comparison_index| index != comparison_index),
                };

                Ok(if comparison_result { Some(index) } else { None })
            }
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        index: NodeIndex,
        operand: &NodeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let arithmetic_index = operand
            .evaluate(medrecord)?
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
    fn evaluate_either_or(
        medrecord: &MedRecord,
        index: NodeIndex,
        either: &Wrapper<NodeIndexOperand>,
        or: &Wrapper<NodeIndexOperand>,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let either_result = either.evaluate(medrecord)?;
        let or_result = or.evaluate(medrecord)?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

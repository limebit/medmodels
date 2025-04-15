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
            attributes::AttributesTreeOperand, nodes::NodeOperand, traits::DeepClone,
            values::MultipleValuesOperand, wrapper::Wrapper, BoxedIterator,
        },
        CardinalityWrapper, EdgeIndex, Group, MedRecordAttribute, MedRecordValue,
    },
    MedRecord,
};
use itertools::Itertools;
use std::{
    collections::HashSet,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone)]
pub enum EdgeOperation {
    Values {
        operand: Wrapper<MultipleValuesOperand<EdgeOperand>>,
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
            Self::Values { operand } => Box::new(Self::evaluate_values(
                medrecord,
                edge_indices,
                operand.clone(),
            )?),
            Self::Attributes { operand } => Box::new(Self::evaluate_attributes(
                medrecord,
                edge_indices,
                operand.clone(),
            )?),
            Self::Indices { operand } => Box::new(Self::evaluate_indices(
                medrecord,
                edge_indices,
                operand.clone(),
            )?),
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
                // TODO: This is a temporary solution. It should be optimized.
                let either_result = either.evaluate(medrecord)?.collect::<HashSet<_>>();
                let or_result = or.evaluate(medrecord)?.collect::<HashSet<_>>();

                Box::new(edge_indices.filter(move |node_index| {
                    either_result.contains(node_index) || or_result.contains(node_index)
                }))
            }
            Self::Exclude { operand } => {
                let result = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

                Box::new(edge_indices.filter(move |node_index| !result.contains(node_index)))
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
        operand: Wrapper<MultipleValuesOperand<EdgeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        Ok(operand.evaluate(medrecord)?.map(|value| value.0))
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
        operand: Wrapper<AttributesTreeOperand<EdgeOperand>>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        Ok(operand.evaluate(medrecord)?.map(|value| value.0))
    }

    #[inline]
    fn evaluate_indices<'a>(
        medrecord: &MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        operand.evaluate(medrecord)
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

            let groups_of_edge = groups_of_edge.collect::<Vec<_>>();

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

            let attributes_of_edge = attributes_of_edge.collect::<Vec<_>>();

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
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let node_indices = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

        Ok(edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.0)
        }))
    }

    #[inline]
    fn evaluate_target_node<'a>(
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<NodeOperand>,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let node_indices = operand.evaluate(medrecord)?.collect::<HashSet<_>>();

        Ok(edge_indices.filter(move |edge_index| {
            let edge_endpoints = medrecord
                .edge_endpoints(edge_index)
                .expect("Edge must exist");

            node_indices.contains(edge_endpoints.1)
        }))
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
    BinaryArithmeticOpration {
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
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
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
        }
    }
}

impl EdgeIndicesOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        match self {
            Self::EdgeIndexOperation { operand } => {
                Self::evaluate_edge_index_operation(medrecord, indices, operand)
            }
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_edge_index_comparison_operation(medrecord, indices, operand, kind)
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_edge_indices_comparison_operation(medrecord, indices, operand, kind)
            }
            Self::BinaryArithmeticOpration { operand, kind } => {
                Ok(Box::new(Self::evaluate_binary_arithmetic_operation(
                    medrecord,
                    indices,
                    operand,
                    kind.clone(),
                )?))
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
        indices: impl Iterator<Item = &'a EdgeIndex>,
    ) -> MedRecordResult<&'a EdgeIndex> {
        indices.max().ok_or(MedRecordError::QueryError(
            "No indices to compare".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_min<'a>(
        indices: impl Iterator<Item = &'a EdgeIndex>,
    ) -> MedRecordResult<&'a EdgeIndex> {
        indices.min().ok_or(MedRecordError::QueryError(
            "No indices to compare".to_string(),
        ))
    }
    #[inline]
    pub(crate) fn get_count<'a>(indices: impl Iterator<Item = &'a EdgeIndex>) -> EdgeIndex {
        indices.count() as EdgeIndex
    }

    #[inline]
    pub(crate) fn get_sum<'a>(indices: impl Iterator<Item = &'a EdgeIndex>) -> EdgeIndex {
        indices.sum()
    }

    #[inline]
    pub(crate) fn get_first<'a>(
        mut indices: impl Iterator<Item = &'a EdgeIndex>,
    ) -> MedRecordResult<&'a EdgeIndex> {
        indices.next().ok_or(MedRecordError::QueryError(
            "No indices to get the first".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_last<'a>(
        indices: impl Iterator<Item = &'a EdgeIndex>,
    ) -> MedRecordResult<&'a EdgeIndex> {
        indices.last().ok_or(MedRecordError::QueryError(
            "No indices to get the first".to_string(),
        ))
    }

    #[inline]
    fn evaluate_edge_index_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = &'a EdgeIndex>,
        operand: &Wrapper<EdgeIndexOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, &'a EdgeIndex>> {
        Ok(match operand.evaluate(medrecord)? {
            Some(_) => Box::new(indices.into_iter()),
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
    fn evaluate_edge_indices_comparison_operation<'a>(
        medrecord: &MedRecord,
        indices: impl Iterator<Item = EdgeIndex> + 'a,
        comparison_operand: &EdgeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        match comparison_operand {
            EdgeIndicesComparisonOperand::Operand(operand) => {
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
            EdgeIndicesComparisonOperand::Indices(comparison_indices) => match kind {
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
        indices: impl Iterator<Item = EdgeIndex>,
        operand: &EdgeIndexComparisonOperand,
        kind: BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = EdgeIndex>> {
        let arithmetic_index = operand
            .evaluate(medrecord)?
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
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = EdgeIndex>,
        either: &Wrapper<EdgeIndicesOperand>,
        or: &Wrapper<EdgeIndicesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, EdgeIndex>> {
        let indices = indices.collect::<Vec<_>>();

        let either_indices = either.evaluate(medrecord)?;
        let or_indices = or.evaluate(medrecord)?;

        Ok(Box::new(either_indices.chain(or_indices).unique()))
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
    BinaryArithmeticOpration {
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
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
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
        }
    }
}

impl EdgeIndexOperation {
    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        index: EdgeIndex,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        match self {
            Self::EdgeIndexComparisonOperation { operand, kind } => {
                Self::evaluate_edge_index_comparison_operation(medrecord, index, operand, kind)
            }
            Self::EdgeIndicesComparisonOperation { operand, kind } => {
                Self::evaluate_edge_indcies_comparison_operation(medrecord, index, operand, kind)
            }
            Self::BinaryArithmeticOpration { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, index, operand, kind)
            }
            Self::EitherOr { either, or } => Self::evaluate_either_or(medrecord, index, either, or),
            Self::Exclude { operand } => {
                let result = operand.evaluate(medrecord)?.is_some();

                Ok(if result { None } else { Some(index) })
            }
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
    fn evaluate_edge_indcies_comparison_operation(
        medrecord: &MedRecord,
        index: EdgeIndex,
        comparison_operand: &EdgeIndicesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        match comparison_operand {
            EdgeIndicesComparisonOperand::Operand(operand) => {
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
            EdgeIndicesComparisonOperand::Indices(comparison_indices) => {
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
        index: EdgeIndex,
        operand: &EdgeIndexComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        let arithmetic_index = operand
            .evaluate(medrecord)?
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
        let either_result = either.evaluate(medrecord)?;
        let or_result = or.evaluate(medrecord)?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}
